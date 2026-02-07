# -*- coding: utf-8 -*-
"""
TF2 + CVXPY 实现 LipSDP-Layer（Theorem 2）计算“已训练部署模型”的 Lipschitz 上界
- 支持加载 SavedModel/.keras 或 checkpoint 权重
- BN 用推理态 moving_mean / moving_variance；Dropout 推理态忽略
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import cvxpy as cp

# -----------------------------
# 0) 你需要：pip install cvxpy
#    强烈建议装 MOSEK（否则用 SCS 可能慢/不稳）
# -----------------------------

# -----------------------------
# 1) 你的模型构建（你需要把这里替换为你真实的 Model 类）
# -----------------------------
def build_your_model(sparse_feature_names,
                     dense_feature_names,
                     sparse_feature_dim,
                     dense_feature_dim,
                     targets,
                     treatment_order):
    """
    这里给一个“结构占位”的示例：你应当用你真实训练时的模型类来构建，
    并确保 model.user_tower 和 model.task_towers[name] 存在且结构一致。
    """
    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.sparse_feature_names = sparse_feature_names
            self.dense_feature_names = dense_feature_names
            self.sparse_feature_dim = sparse_feature_dim
            self.dense_feature_dim = dense_feature_dim
            self.targets = targets
            self.treatment_order = treatment_order

            # user_tower（与你给的一致）
            self.user_tower = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
            ], name='user_tower')

            # 提前 build 以创建权重（你原代码也这么做）
            user_tower_input_dim = (
                len(self.sparse_feature_names) * self.sparse_feature_dim +
                len(self.dense_feature_names) * self.dense_feature_dim
            )
            self.user_tower.build((None, user_tower_input_dim))

            # task towers
            self.task_towers = {}
            tower_dims = [64, 32, 1]
            for target in self.targets:
                for treatment in self.treatment_order:
                    name = f"{target}_treatment_{treatment}_tower"
                    self.task_towers[name] = tf.keras.Sequential(
                        [tf.keras.layers.Dense(d, activation='relu', kernel_initializer='glorot_normal')
                         for d in tower_dims[:-1]] +
                        [tf.keras.layers.Dense(tower_dims[-1], kernel_initializer='glorot_normal')],
                        name=name
                    )
                    # task tower 输入是 user_tower 最后维度 128
                    self.task_towers[name].build((None, 128))

        def call(self, x, training=False):
            h = self.user_tower(x, training=training)
            # 这里只返回一个 tower 的输出示意（你的真实模型可不同）
            first_name = list(self.task_towers.keys())[0]
            y = self.task_towers[first_name](h, training=training)
            return y

    return MyModel()


# -----------------------------
# 2) 工具：谱范数 baseline（可选）
# -----------------------------
def spectral_norm(W: np.ndarray) -> float:
    s = np.linalg.svd(W, compute_uv=False)
    return float(s[0])

def product_spectral_norm(weights):
    val = 1.0
    for W in weights:
        val *= spectral_norm(W)
    return val


# -----------------------------
# 3) BN 推理态折叠：y = gamma*(x-mean)/sqrt(var+eps)+beta = D x + c
# -----------------------------
def _bn_affine_params(bn: tf.keras.layers.BatchNormalization):
    gamma = bn.gamma.numpy() if bn.gamma is not None else np.ones((bn.axis[-1],), dtype=np.float32)
    beta  = bn.beta.numpy()  if bn.beta  is not None else np.zeros((bn.axis[-1],), dtype=np.float32)
    mean  = bn.moving_mean.numpy()
    var   = bn.moving_variance.numpy()
    eps   = bn.epsilon
    scale = gamma / np.sqrt(var + eps)
    shift = beta - scale * mean
    return scale.astype(np.float64), shift.astype(np.float64)


# -----------------------------
# 4) 从 user_tower 提取线性链（Dense+ReLU, BN, Dropout）并折叠 BN
# -----------------------------
def extract_linear_chain_from_user_tower(user_tower: tf.keras.Sequential):
    layers = user_tower.layers
    dense_indices = [i for i, l in enumerate(layers) if isinstance(l, tf.keras.layers.Dense)]
    bn_indices    = [i for i, l in enumerate(layers) if isinstance(l, tf.keras.layers.BatchNormalization)]

    if not dense_indices:
        raise ValueError("user_tower 中没有 Dense 层。")

    dense_W = []
    dense_b = []
    hidden_dims = []

    for idx in dense_indices:
        W, b = layers[idx].get_weights()     # Keras Dense: W (in, out)
        W = W.T.astype(np.float64)          # 转 (out, in)
        b = b.astype(np.float64).reshape(-1)
        dense_W.append(W)
        dense_b.append(b)
        hidden_dims.append(W.shape[0])

    input_dim = dense_W[0].shape[1]

    # 折叠 BN -> 下一层 Dense 输入侧
    W_eff = [W.copy() for W in dense_W]
    b_eff = [b.copy() for b in dense_b]

    for bn_i in bn_indices:
        bn = layers[bn_i]
        scale, shift = _bn_affine_params(bn)

        # 找 bn 后最近的 Dense
        next_dense_pos = None
        for j in dense_indices:
            if j > bn_i:
                next_dense_pos = j
                break
        if next_dense_pos is None:
            continue
        next_k = dense_indices.index(next_dense_pos)

        Wn_old = W_eff[next_k]
        if Wn_old.shape[1] != scale.shape[0]:
            raise ValueError(
                f"BN 输出维度 {scale.shape[0]} 与下一层 Dense 输入维度 {Wn_old.shape[1]} 不匹配，无法折叠。"
            )
        W_eff[next_k] = Wn_old @ np.diag(scale)
        b_eff[next_k] = b_eff[next_k] + (Wn_old @ shift)

    # Lipschitz 上界只与线性映射的算子范数有关，偏置不影响全局 Lipschitz
    return W_eff, hidden_dims, input_dim


# -----------------------------
# 5) 从 task_tower 提取线性链（Dense+ReLU... + 最后一层线性）
# -----------------------------
def extract_linear_chain_from_task_tower(task_tower: tf.keras.Sequential, input_dim: int):
    dense_layers = [l for l in task_tower.layers if isinstance(l, tf.keras.layers.Dense)]
    if not dense_layers:
        raise ValueError("task_tower 中没有 Dense 层。")

    Ws = []
    hidden_dims = []
    for k, layer in enumerate(dense_layers):
        W, b = layer.get_weights()
        W = W.T.astype(np.float64)  # (out, in)
        Ws.append(W)
        if k < len(dense_layers) - 1:
            hidden_dims.append(W.shape[0])

    if Ws[0].shape[1] != input_dim:
        raise ValueError(f"task_tower 第一层输入维度 {Ws[0].shape[1]} != 期望 {input_dim}")

    return Ws, hidden_dims, Ws[-1].shape[0]


# -----------------------------
# 6) LipSDP-Layer（Theorem 2）SDP：min rho s.t. LMI <= 0,  L = sqrt(rho*)
# -----------------------------
def lipsdp_layer_bound(weights, hidden_dims, input_dim, solver="MOSEK", verbose=False):
    # ReLU 属于 slope-restricted on [0,1]
    alpha, beta = 0.0, 1.0

    L = len(hidden_dims)            # ReLU 层数
    assert len(weights) == L + 1, "weights 数量应为 ReLU层数+1（最后一层输出线性）"

    n_total = int(sum(hidden_dims))
    total_x = int(input_dim + n_total)  # x=[x0;x1;...;xL]

    rho = cp.Variable(nonneg=True)
    lambdas = cp.Variable(L, nonneg=True)

    def blkdiag(blocks):
        rows = []
        for i, Bi in enumerate(blocks):
            row = []
            for j, Bj in enumerate(blocks):
                if i == j:
                    row.append(Bi)
                else:
                    row.append(cp.Constant(np.zeros((Bi.shape[0], Bj.shape[1]))))
            rows.append(row)
        return cp.bmat(rows)

    T_blocks = [lambdas[k] * np.eye(hidden_dims[k]) for k in range(L)]
    T = blkdiag(T_blocks)  # (n_total, n_total)

    # 构造 A, B（论文里把 preact 与 postact 关系写成二次约束）
    A = np.zeros((n_total, total_x), dtype=np.float64)
    B = np.zeros((n_total, total_x), dtype=np.float64)

    seg_starts = [0]
    cur = input_dim
    for nk in hidden_dims:
        seg_starts.append(cur)
        cur += nk

    row_cursor = 0
    for k in range(1, L + 1):
        nk = hidden_dims[k - 1]
        rows = slice(row_cursor, row_cursor + nk)

        prev_dim = input_dim if k == 1 else hidden_dims[k - 2]
        col_prev = slice(seg_starts[k - 1], seg_starts[k - 1] + prev_dim)

        W_prev = weights[k - 1]  # (nk, prev_dim)
        A[rows, col_prev] = W_prev

        col_k = slice(seg_starts[k], seg_starts[k] + nk)
        B[rows, col_k] = np.eye(nk, dtype=np.float64)
        row_cursor += nk

    # 输出矩阵 C（只在 xL 位置放 W_out）
    W_out = weights[-1]
    nout = W_out.shape[0]
    C = np.zeros((nout, total_x), dtype=np.float64)
    start_xL = seg_starts[L]
    C[:, start_xL:start_xL + hidden_dims[L - 1]] = W_out

    # ReLU multiplier: [[-2abT,(a+b)T],[(a+b)T,-2T]]，a=0,b=1 => [[0,T],[T,-2T]]
    P = cp.bmat([
        [cp.Constant(np.zeros((n_total, n_total))), T],
        [T, -2.0 * T]
    ])

    AB = np.vstack([A, B])           # (2n_total, total_x)
    term1 = AB.T @ P @ AB

    # term2: -rho on x0 block + W_out^T W_out on xL block
    term2 = cp.Constant(np.zeros((total_x, total_x)))
    term2 = term2 + cp.bmat([
        [-rho * np.eye(input_dim, dtype=np.float64),
         cp.Constant(np.zeros((input_dim, total_x - input_dim), dtype=np.float64))],
        [cp.Constant(np.zeros((total_x - input_dim, input_dim), dtype=np.float64)),
         cp.Constant(np.zeros((total_x - input_dim, total_x - input_dim), dtype=np.float64))]
    ])

    add_mat = np.zeros((total_x, total_x), dtype=np.float64)
    add_mat[start_xL:start_xL + hidden_dims[L - 1],
            start_xL:start_xL + hidden_dims[L - 1]] = (W_out.T @ W_out)
    term2 = term2 + cp.Constant(add_mat)

    M = term1 + term2
    constraints = [M << 0]

    prob = cp.Problem(cp.Minimize(rho), constraints)

    if solver.upper() == "MOSEK":
        prob.solve(solver=cp.MOSEK, verbose=verbose)
    elif solver.upper() == "SCS":
        prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-5, max_iters=200000)
    else:
        prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP 未求到最优解，status={prob.status}")

    rho_opt = float(rho.value)
    L_bound = float(np.sqrt(max(rho_opt, 0.0)))
    return L_bound, rho_opt, np.array(lambdas.value).reshape(-1)


# -----------------------------
# 7) “部署模型”一键：加载已训练权重 -> 推理态 -> 取 tower -> LipSDP
# -----------------------------
def lipschitz_bound_for_deployed_model(model,
                                       task_tower_name: str,
                                       solver="MOSEK",
                                       verbose=False):
    # 确保该 tower 存在
    if not hasattr(model, "user_tower"):
        raise AttributeError("model 缺少 user_tower 属性。")
    if not hasattr(model, "task_towers") or task_tower_name not in model.task_towers:
        raise AttributeError(f"model.task_towers 不存在或不含 {task_tower_name}")

    user_tower = model.user_tower
    task_tower = model.task_towers[task_tower_name]

    # 提取线性链 + BN 折叠
    user_W, user_hidden_dims, user_in_dim = extract_linear_chain_from_user_tower(user_tower)

    # task tower 输入维度 = user 最后一层维度
    task_in_dim = user_hidden_dims[-1]
    task_W, task_hidden_dims, out_dim = extract_linear_chain_from_task_tower(task_tower, input_dim=task_in_dim)

    weights = user_W + task_W
    hidden_dims = user_hidden_dims + task_hidden_dims
    input_dim = user_in_dim

    baseline = product_spectral_norm(weights)
    print(f"[Baseline] spectral-norm product ≈ {baseline:.6f}")

    L_bound, rho_opt, lambdas = lipsdp_layer_bound(
        weights=weights,
        hidden_dims=hidden_dims,
        input_dim=input_dim,
        solver=solver,
        verbose=verbose
    )
    print(f"[LipSDP-Layer] rho*={rho_opt:.6f}, L=√rho*={L_bound:.6f}")
    print(f"[LipSDP-Layer] lambdas={lambdas}")
    return L_bound


# -----------------------------
# 8) 加载训练好模型：两种方式
# -----------------------------
def load_trained_model_via_savedmodel(model_path: str):
    # SavedModel/.keras：包含结构+权重（推荐最省事）
    # 参考 TF 官方 Save/Load 教程 :contentReference[oaicite:2]{index=2}
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def load_trained_model_via_checkpoint(ckpt_path: str, model_builder, dummy_input_dim: int):
    """
    ckpt: 仅权重 -> 需要先用相同结构 build 出 model，再 load_weights
    TF 官方说明：save() vs save_weights() 不同 :contentReference[oaicite:3]{index=3}
    """
    model = model_builder()
    # 触发变量创建（BN moving stats / Dense weights 都要先建好）
    _ = model(tf.zeros((1, dummy_input_dim), dtype=tf.float32), training=False)
    model.load_weights(ckpt_path)
    # 再跑一次推理态，确保所有层处于推理逻辑（BN 用 moving stats）
    _ = model(tf.zeros((1, dummy_input_dim), dtype=tf.float32), training=False)
    return model


# -----------------------------
# 9) 主程序示例：按你的配置替换即可
# -----------------------------
if __name__ == "__main__":
    # ====== A) 你的真实配置（示例占位）======
    SPARSE_FEATURE_NAME = ['feat1_seq1', 'feat1_seq2', 'feat1_seq3', 'feat1_seq4', 'feat1_seq5', 'feat1_seq6', 'feat1_seq7', 'feat1_seq8', 'feat2_seq1']
    DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15']
    sparse_feature_dim = 8
    dense_feature_dim = 1
    targets = ['paid', 'cost']
    treatment_order = [0, 1]

    user_tower_input_dim = (
        len(SPARSE_FEATURE_NAME) * sparse_feature_dim +
        len(DENSE_FEATURE_NAME) * dense_feature_dim
    )

    # 你要算哪个 tower 的 Lipschitz 上界（示例）
    task_tower_name = f"{targets[0]}_treatment_{treatment_order[0]}_tower"

    # ====== B) 选择一种加载方式 ======
    # 方式1：SavedModel 或 .keras
    MODEL_PATH = "../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40"  # e.g. "/path/to/saved_model_or_model.keras"
    # # 方式2：Checkpoint（只存权重）
    # CKPT_PATH = ""   # e.g. "/path/to/ckpt"

    # ====== C) 构建/加载 ======
    model = load_trained_model_via_savedmodel(MODEL_PATH)

    # if MODEL_PATH and os.path.exists(MODEL_PATH):
    # else:
    #     def builder():
    #         return build_your_model(
    #             SPARSE_FEATURE_NAME, DENSE_FEATURE_NAME,
    #             sparse_feature_dim, dense_feature_dim,
    #             targets, treatment_order
    #         )
    #     if not (CKPT_PATH and os.path.exists(CKPT_PATH)):
    #         raise FileNotFoundError("请设置 MODEL_PATH 或 CKPT_PATH 为真实已训练模型路径。")
    #     model = load_trained_model_via_checkpoint(CKPT_PATH, builder, dummy_input_dim=user_tower_input_dim)

    # ====== D) 计算部署模型 Lipschitz 上界 ======
    # solver 推荐 MOSEK；没有的话用 SCS（可能更慢/更松）
    L = lipschitz_bound_for_deployed_model(
        model=model,
        task_tower_name=task_tower_name,
        solver="SCS",
        verbose=False
    )
    print("Final deployed-model Lipschitz upper bound:", L)
