import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cvxpy as cp
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union
import json


def _try_solve(prob: cp.Problem, prefer: Optional[str] = None, verbose: bool = False) -> None:
    """
    Try solving with a preferred solver; fallback to others commonly available.
    """
    solvers = []
    if prefer:
        solvers.append(prefer)

    # Common CVXPY solvers (availability depends on environment)
    solvers += ["MOSEK", "GUROBI", "CVXOPT", "SCS", "ECOS"]

    last_err = None
    for s in solvers:
        try:
            prob.solve(solver=getattr(cp, s), verbose=verbose)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                # print(f"solver: {s}")
                return
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"SDP solve failed. Last error: {last_err}, status={prob.status}")

# -----------------------------
# Empirical Lipschitz estimate
# -----------------------------
def _as_numpy_features(x: Any) -> Any:
    """
    Convert feature batch to numpy containers (np.ndarray or dict of np.ndarray).
    Keeps dtype if possible.
    """
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if tf.is_tensor(v):
                out[k] = v.numpy()
            else:
                out[k] = np.asarray(v)
        return out
    else:
        if tf.is_tensor(x):
            return x.numpy()
        return np.asarray(x)


def _slice_one(x_batch_np: Any, idx: int) -> Any:
    """
    Take a single sample (batch size 1) from numpy features.
    Supports np.ndarray or dict[str, np.ndarray].
    """
    if isinstance(x_batch_np, dict):
        x1 = {}
        for k, v in x_batch_np.items():
            # assume first dim is batch
            x1[k] = v[idx:idx+1]
        return x1
    else:
        return x_batch_np[idx:idx+1]


def _l2_norm_features(delta: Any) -> float:
    """
    L2 norm for tensor input or dict input (concatenate all flattened parts).
    """
    if isinstance(delta, dict):
        s = 0.0
        for v in delta.values():
            vv = np.asarray(v)
            s += float(np.sum(vv.astype(np.float64).ravel() ** 2))
        return float(np.sqrt(s))
    else:
        d = np.asarray(delta)
        return float(np.linalg.norm(d.reshape(-1).astype(np.float64)))


def _randn_like_features(x1: Any, rng: np.random.Generator) -> Any:
    """
    Random normal noise with the same structure/shape as x1.
    Only perturb float tensors/arrays; for non-float (e.g., int sparse ids),
    we keep delta = 0 to avoid invalid inputs.
    """
    if isinstance(x1, dict):
        delta = {}
        for k, v in x1.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.floating):
                delta[k] = rng.standard_normal(size=arr.shape).astype(arr.dtype)
            else:
                delta[k] = np.zeros_like(arr)
        return delta
    else:
        arr = np.asarray(x1)
        if not np.issubdtype(arr.dtype, np.floating):
            # if input isn't float, we cannot add meaningful noise safely
            return np.zeros_like(arr)
        return rng.standard_normal(size=arr.shape).astype(arr.dtype)


def _scale_delta_to_epsilon(delta: Any, epsilon: float) -> Any:
    """
    Scale delta so that ||delta||_2 = epsilon (if norm > 0).
    """
    norm = _l2_norm_features(delta)
    if norm <= 0.0:
        return delta
    scale = float(epsilon) / norm

    if isinstance(delta, dict):
        return {k: (np.asarray(v) * scale).astype(np.asarray(v).dtype) for k, v in delta.items()}
    else:
        d = np.asarray(delta)
        return (d * scale).astype(d.dtype)


def _add_delta(x1: Any, delta: Any) -> Any:
    """
    x1 + delta for tensor input or dict input.
    Keeps dtype and does not touch non-float parts (they should have delta=0).
    """
    if isinstance(x1, dict):
        out = {}
        for k in x1.keys():
            xv = np.asarray(x1[k])
            dv = np.asarray(delta[k])
            if np.issubdtype(xv.dtype, np.floating):
                out[k] = (xv + dv).astype(xv.dtype)
            else:
                out[k] = xv  # unchanged
        return out
    else:
        x = np.asarray(x1)
        d = np.asarray(delta)
        if np.issubdtype(x.dtype, np.floating):
            return (x + d).astype(x.dtype)
        return x


def empirical_lipschitz_estimate(
    model: tf.keras.Model,
    x_batch: Any,
    n_samples: int = 1000,
    epsilon: float = 0.01,
    seed: int = 1234,
) -> float:
    """
    通过随机扰动估计经验 Lipschitz 常数（L2 输入扰动，L2 输出差异）。
    - 支持 x_batch 为 np.ndarray / tf.Tensor
    - 支持 x_batch 为 dict[str, np.ndarray/tf.Tensor]（常见于多输入模型）
    - 对非浮点输入（如 sparse id int）默认不扰动，避免输入非法

    输出差异使用 RobustnessSDP.compute_utilities_from_predictions() 得到的 U（效用矩阵），
    然后计算 ||U(x)-U(x+delta)||_2 / ||delta||_2 的最大值。
    """
    x_batch_np = _as_numpy_features(x_batch)

    # batch size
    if isinstance(x_batch_np, dict):
        any_key = next(iter(x_batch_np.keys()))
        B = int(x_batch_np[any_key].shape[0])
    else:
        B = int(np.asarray(x_batch_np).shape[0])

    rng = np.random.default_rng(seed)
    max_ratio = 0.0

    for _ in range(int(n_samples)):
        idx = int(rng.integers(low=0, high=B))
        x1 = _slice_one(x_batch_np, idx)

        # random direction (same structure as x1)
        delta = _randn_like_features(x1, rng)
        delta = _scale_delta_to_epsilon(delta, float(epsilon))

        x2 = _add_delta(x1, delta)

        # forward (training=False)
        pred1 = model(x1, training=False)
        pred2 = model(x2, training=False)

        # utilities (U: (1,2) or (1,K))
        u1, _ = RobustnessSDP.compute_utilities_from_predictions(pred1)
        u2, _ = RobustnessSDP.compute_utilities_from_predictions(pred2)

        output_diff = float(np.linalg.norm((u1 - u2).reshape(-1).astype(np.float64)))
        input_diff = float(_l2_norm_features(delta))

        if input_diff > 0:
            ratio = output_diff / input_diff
            if ratio > max_ratio:
                max_ratio = ratio

    return float(max_ratio)


class RobustnessSDP:
    """
    针对 Multi-tower 结构的 SDP Lipschitz 验证工具
    """

    @staticmethod
    def verify_dense_layer_lipschitz_sdp(
        W: np.ndarray,
        p_structure: str = "diag",
        solver: Optional[str] = None,
        eps_pd: float = 1e-6, 
        verbose: bool = False
    ) -> float:
        """
        计算单层 Dense 权重的谱范数上界。
        修正：强制 P >= I，消除数值不稳定性。
        """
        if W.ndim != 2:
            return 1.0 # 兜底，非2D权重（如Bias）不影响Lipschitz
        
        n_out, n_in = W.shape
        gamma = cp.Variable(nonneg=True)

        # 1. 构造 P
        if p_structure.lower() == "full":
            P = cp.Variable((n_in, n_in), PSD=True)
            P_mat = P
            # 约束 P 的最小特征值 >= 1 (代替 epsilon)
            P_lb = eps_pd * np.eye(n_in) 
        else: # diag
            p = cp.Variable(n_in, nonneg=True)
            P_mat = cp.diag(p)
            P_lb = eps_pd * np.eye(n_in)

        # 2. LMI 约束: 
        # [ P       P W^T ]
        # [ W P    gamma I]  >= 0
        # 且 P >= I
        block = cp.bmat([
            [P_mat,             P_mat @ W.T],
            [W @ P_mat,         gamma * np.eye(n_out)],
        ])

        constraints = [block >> 0, P_mat - P_lb >> 0]
        
        prob = cp.Problem(cp.Minimize(gamma), constraints)
        _try_solve(prob, prefer=solver, verbose=verbose)

        if gamma.value is None:
            raise RuntimeError(f"SDP returned no gamma. status={prob.status}")

        # try:
        #     # 优先尝试 SCS 或 MOSEK，如果失败则捕获
        #     prob.solve(solver=solver, verbose=verbose)
        # except Exception as e:
        #     print(f"  [SDP Error] {e}. Fallback to numpy spectral norm.")
        #     return float(np.linalg.norm(W, ord=2))

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or gamma.value is None:
            # 求解失败兜底
            return float(np.linalg.norm(W, ord=2))

        # 3. 计算结果
        # 原理：W P W^T <= gamma I. 因为 P >= I，所以 W W^T <= gamma I.
        # L = sqrt(gamma)
        return float(np.sqrt(max(gamma.value, 1e-16) / eps_pd))

    @staticmethod
    def get_sequential_lipschitz(model_seq: tf.keras.Sequential) -> float:
        """
        计算一个串行模块（Sequential）的总 Lipschitz 常数。
        L_total = L_layer1 * L_layer2 * ...
        """
        L_total = 1.0
        
        # 遍历层，计算每一层的 L
        for layer in model_seq.layers:
            L_layer = 1.0
            
            # --- Dense ---
            if isinstance(layer, tf.keras.layers.Dense):
                kernel = layer.kernel.numpy() # (n_in, n_out)
                W = kernel.T                  # (n_out, n_in) 为 SDP 格式
                
                # 1. 线性部分的 L
                L_lin = RobustnessSDP.verify_dense_layer_lipschitz_sdp(W, p_structure="diag")
                
                # 2. 激活函数的 L
                act_name = getattr(layer.activation, "__name__", "linear").lower()
                if "relu" in act_name or "linear" in act_name:
                    L_act = 1.0
                elif "sigmoid" in act_name:
                    L_act = 0.25
                elif "tanh" in act_name:
                    L_act = 1.0
                else:
                    L_act = 1.0 # 保守估计
                
                L_layer = L_lin * L_act
                print(f"    - Layer {layer.name:<20} (Dense): L_lin={L_lin:.4f} * L_act={L_act} = {L_layer:.4f}")

            # --- BatchNorm ---
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                # BN 在推理时是线性缩放: scale = gamma / sqrt(var + eps)
                # L = max(|scale|)
                mv = layer.moving_variance.numpy()
                eps = layer.epsilon
                if layer.scale:
                    gamma = layer.gamma.numpy()
                else:
                    gamma = np.ones_like(mv)
                
                scale = np.abs(gamma / np.sqrt(mv + eps))
                L_layer = float(np.max(scale))
                print(f"    - Layer {layer.name:<20} (BN)   : L={L_layer:.4f}")

            # --- Dropout ---
            elif isinstance(layer, tf.keras.layers.Dropout):
                # 推理模式下 Dropout 是 Identity，L=1
                L_layer = 1.0
            
            else:
                # 其他层默认 L=1 (如 Reshape, Flatten)
                pass

            L_total *= L_layer

        return L_total

    @staticmethod
    def analyze_multitower_lipschitz(model, treatment_order, targets):
        """
        针对你的 Multi-tower 结构进行解析
        """
        print("\n[SDP Analysis] Calculating Lipschitz for User Tower (Shared)...")
        # 1. 计算 Shared Bottom (User Tower) 的 L
        # 注意：这里假设 model.user_tower 是一个 Sequential 对象
        try:
            user_tower = model.user_tower
        except AttributeError:
            # 如果加载的是 SavedModel，属性可能丢失，需要通过 get_layer 获取
            print("  [Warn] model.user_tower attribute missing. Trying by name 'user_tower'...")
            user_tower = model.get_layer('user_tower')

        L_shared = RobustnessSDP.get_sequential_lipschitz(user_tower)
        print(f"  => L_shared (User Tower) = {L_shared:.6f}")

        # 2. 计算各个 Head 的 L，并结合 Shared L
        tower_lipschitz_map = {}
        
        # 获取所有可能的任务名
        task_names = []
        for target in targets:
            for treatment in treatment_order:
                name = "{}_treatment_{}_tower".format(target, treatment)
                task_names.append(name)
        
        print("\n[SDP Analysis] Calculating Lipschitz for Task Towers (Heads)...")
        for name in task_names:
            try:
                # 尝试从属性获取，或从层名获取
                if hasattr(model, 'task_towers') and name in model.task_towers:
                    tower = model.task_towers[name]
                else:
                    tower = model.get_layer(name)
                
                print(f"  Analysing {name}...")
                L_head = RobustnessSDP.get_sequential_lipschitz(tower)
                
                # 路径总 L = Shared * Head
                L_path = L_shared * L_head
                tower_lipschitz_map[name] = L_path
                print(f"  => L_path ({name}) = {L_shared:.4f} * {L_head:.4f} = {L_path:.6f}")
                
            except Exception as e:
                print(f"  [Error] Could not find or analyze tower '{name}': {e}")
                tower_lipschitz_map[name] = float('inf')

        return L_shared, tower_lipschitz_map

    @staticmethod
    def calculate_decision_lipschitz(tower_lipschitz_map, targets, lambda_cost=0.5):
        """
        计算最终决策函数 U(x) 的 Lipschitz 常数。
        Utility U = (Paid1 - Paid0) - lambda * (Cost1 - Cost0)
        Lipschitz(Sum) <= Sum(Lipschitz)
        """
        # 找到对应的 Key
        p1 = tower_lipschitz_map.get(f"{targets[0]}_treatment_1_tower", 0.0) # 假设 targets[0] 是 paid
        p0 = tower_lipschitz_map.get(f"{targets[0]}_treatment_0_tower", 0.0)
        c1 = tower_lipschitz_map.get(f"{targets[1]}_treatment_1_tower", 0.0) # 假设 targets[1] 是 cost
        c0 = tower_lipschitz_map.get(f"{targets[1]}_treatment_0_tower", 0.0)

        # L_decision <= L_p1 + L_p0 + |lambda|*L_c1 + |lambda|*L_c0
        L_decision = p1 + p0 + abs(lambda_cost) * c1 + abs(lambda_cost) * c0
        return L_decision
    
    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if tf.is_tensor(x):
            return x.numpy()
        return np.asarray(x)
    
    @staticmethod
    def compute_utilities_from_predictions(
        preds: Any,
        paid_prefix: str = "paid_treatment_",
        cost_prefix: str = "cost_treatment_",
        lambda_cost: float = 0.5,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Binary decision utility:
        u1 = (paid_1 - paid_0) - lambda * (cost_1 - cost_0)
        decide treatment=1 if u1 > 0 else 0

        To keep the rest of the pipeline unchanged (argmax & margin),
        we build U = [u0, u1] with u0 = 0. So argmax(U) matches the threshold rule.

        Expected keys in preds:
        paid_treatment_0, paid_treatment_1, cost_treatment_0, cost_treatment_1

        Returns:
        U: shape (B, 2) where columns correspond to treatment_ids [0, 1]
        treatment_ids: [0, 1]
        """
        if not isinstance(preds, dict):
            raise ValueError("Model output must be a dict for compute_utilities_from_predictions().")

        # Required keys
        k_p0 = f"{paid_prefix}0"
        k_p1 = f"{paid_prefix}1"
        k_c0 = f"{cost_prefix}0"
        k_c1 = f"{cost_prefix}1"

        missing = [k for k in [k_p0, k_p1, k_c0, k_c1] if k not in preds]
        if missing:
            raise ValueError(
                "Missing required prediction keys for incremental decision: " + ", ".join(missing)
            )

        paid0 = RobustnessSDP._to_numpy(preds[k_p0]).reshape(-1)
        paid1 = RobustnessSDP._to_numpy(preds[k_p1]).reshape(-1)
        cost0 = RobustnessSDP._to_numpy(preds[k_c0]).reshape(-1)
        cost1 = RobustnessSDP._to_numpy(preds[k_c1]).reshape(-1)

        delta_paid = paid1 - paid0
        delta_cost = cost1 - cost0

        u1 = delta_paid - float(lambda_cost) * delta_cost
        u0 = np.zeros_like(u1)

        U = np.stack([u0, u1], axis=1)  # (B, 2)
        treatment_ids = [0, 1]
        return U, treatment_ids
    
import matplotlib.pyplot as plt
def verify_decision_robustness(model, sample_features, L_decision, lambda_cost=0.5, epsilon=0.1):
    """
    计算决策边界的鲁棒性
    :param model: 训练好的 Keras 模型
    :param sample_features: 输入特征 (Dict[str, Tensor] 或 Tensor)
    :param L_decision: 上一步计算出的决策函数 Lipschitz 总常数 (L_p1 + L_p0 + L_c1 + L_c0)
    :param lambda_cost: 效用公式中 Cost 的权重
    :param epsilon: 我们想要验证的攻击/扰动半径 (L2 norm)
    """
    
    print("\n========== 4. 决策鲁棒性验证 (Decision Robustness) ==========")
    print(f"[配置] Epsilon (扰动半径): {epsilon}")
    print(f"[配置] Lambda Cost: {lambda_cost}")
    print(f"[参数] L_decision (Diff Lipschitz): {L_decision:.6f}")

    # ---------------------------------------------------------
    # 4.1 模型推理 (Inference)
    # ---------------------------------------------------------
    print("正在进行模型推理...")
    # training=False 确保 BatchNorm 使用移动平均统计量，Dropout 关闭
    preds = model(sample_features, training=False)

    # 将 Tensor 转为 Numpy 以便计算
    def to_np(x):
        return x.numpy().flatten() if tf.is_tensor(x) else np.asarray(x).flatten()

    # 提取预测值 (根据你的命名习惯调整 Key)
    try:
        p0 = to_np(preds['paid_treatment_0_tower'])
        p1 = to_np(preds['paid_treatment_1_tower'])
        c0 = to_np(preds['cost_treatment_0_tower'])
        c1 = to_np(preds['cost_treatment_1_tower'])
    except KeyError:
        # 兼容旧的命名方式 (如果没有 _tower 后缀)
        p0 = to_np(preds.get('paid_treatment_0', np.zeros(1)))
        p1 = to_np(preds.get('paid_treatment_1', np.zeros(1)))
        c0 = to_np(preds.get('cost_treatment_0', np.zeros(1)))
        c1 = to_np(preds.get('cost_treatment_1', np.zeros(1)))
        print("[Warn] 使用了备用 Key (不带 _tower 后缀)")

    # ---------------------------------------------------------
    # 4.2 计算效用与 Margin
    # Utility = Paid - lambda * Cost
    # Delta U = U(treatment=1) - U(treatment=0)
    # ---------------------------------------------------------
    u0 = p0 - lambda_cost * c0
    u1 = p1 - lambda_cost * c1
    
    # 决策逻辑：如果 diff_u > 0，选择 T=1；否则 T=0
    diff_u = u1 - u0
    
    # Margin 是距离决策边界 (0) 的距离
    margin = np.abs(diff_u)
    
    # 原始决策 (Original Decision)
    original_decision = (diff_u > 0).astype(int)

    # ---------------------------------------------------------
    # 4.3 计算安全半径 (Safe Radius)
    # 理论：如果 Margin(x) > L_decision * ||delta||，则决策不会翻转
    # Safe Radius = Margin(x) / L_decision
    # ---------------------------------------------------------
    # 防止除以 0
    valid_L = max(L_decision, 1e-8)
    
    # 注意：这里不需要 * 2，因为 L_decision 已经是 (L_p1 + L_p0 + ...) 的和
    # 它直接约束了 diff_u 函数的变化率
    safe_radius = margin / valid_L

    # 判断是否鲁棒
    is_robust = safe_radius >= epsilon

    # ---------------------------------------------------------
    # 4.4 统计报告
    # ---------------------------------------------------------
    n_samples = len(margin)
    robust_count = np.sum(is_robust)
    robust_ratio = robust_count / n_samples

    print("\n[验证结果报告]")
    print(f"样本总数          : {n_samples}")
    print(f"鲁棒样本数        : {robust_count}")
    print(f"鲁棒比例 (Robust%) : {robust_ratio * 100:.2f}%")
    print(f"-" * 30)
    print(f"平均 Margin       : {np.mean(margin):.6f}")
    print(f"平均 Safe Radius  : {np.mean(safe_radius):.6f}")
    print(f"最小 Safe Radius  : {np.min(safe_radius):.6f}")
    
    # ---------------------------------------------------------
    # 4.5 (可选) 可视化分布
    # ---------------------------------------------------------
    # 如果样本量大，画图看看 Margin 分布很有帮助
    # try:
    #     plt.figure(figsize=(10, 4))
        
    #     plt.subplot(1, 2, 1)
    #     plt.hist(margin, bins=50, color='skyblue', alpha=0.7, label='Margin')
    #     plt.axvline(x=valid_L * epsilon, color='red', linestyle='--', label=f'Threshold (L*eps)')
    #     plt.title('Decision Margin Distribution')
    #     plt.xlabel('Margin |U1 - U0|')
    #     plt.legend()

    #     plt.subplot(1, 2, 2)
    #     plt.hist(safe_radius, bins=50, color='lightgreen', alpha=0.7, label='Safe Radius')
    #     plt.axvline(x=epsilon, color='red', linestyle='--', label=f'Target Epsilon ({epsilon})')
    #     plt.title('Safe Radius Distribution')
    #     plt.xlabel('Radius (eps)')
    #     plt.legend()
        
    #     plt.tight_layout()
    #     plt.show()
    #     print("[Info] 直方图已生成")
    # except Exception as e:
    #     print(f"[Info] 跳过绘图: {e}")

    return {
        "robust_ratio": robust_ratio,
        "mean_safe_radius": np.mean(safe_radius),
        "is_robust_indices": is_robust,
        "margins": margin
    }
    
# ----------------------------------------------------
# 你的主程序部分
# ----------------------------------------------------
def main_old():
    print("[INFO] Loading model")
    model = tf.keras.models.load_model(
        "../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40",
        compile=False
    )
    model.summary()
    # 假设你已经加载了模型
    # model = ... (加载代码)
    # 如果你是 subclass model，确保你能访问到 .user_tower 属性
    # 如果是 load_model 加载的 saved_model，可能需要用 model.get_layer('user_tower')
    
    # 模拟配置 (根据你的实际配置修改)
    targets = ['paid', 'cost']  # 确保顺序正确
    treatment_order = [0, 1]
    
    print("========== 开始 Multi-tower Lipschitz 验证 ==========")
    
    # 1. 计算各部分 Lipschitz
    L_shared, tower_map = RobustnessSDP.analyze_multitower_lipschitz(
        model, treatment_order, targets
    )
    
    # 2. 计算决策边界的 Lipschitz
    # 假设 Utility = (Paid1 - Paid0) - 0.5 * (Cost1 - Cost0)
    L_decision = RobustnessSDP.calculate_decision_lipschitz(
        tower_map, targets, lambda_cost=0.5
    )
    
    print("\n========== 最终结果 ==========")
    print(f"Shared Tower L : {L_shared:.6f}")
    print(f"决策函数 L_dec : {L_decision:.6f} (用于计算 Margin 的 Robustness)")
    
    # 3. 验证鲁棒性
    # 只有当 L_decision 处于合理范围 (0.1 ~ 50 左右) 时，结果才可信
    if L_decision < 1e-4:
        print("[警告] L_decision 依然极小，请检查：")
        print("1. 模型权重是否因为正则化过强而接近 0？")
        print("2. 是否使用了 Sigmoid 激活函数？(Sigmoid 会导致梯度消失)")
    
    # 4. 计算 Margin
    # 这里的 sample_features 需要你自己提供
    # preds = model(sample_features, training=False)
    # 计算 utility margin...
    # safe_radius = margin / (2 * L_decision)
    
    # 你的验证逻辑...
        
    sample_path = "sample_features_ECLIFT.npz"
    # sample_path = "sample_features_criteo.npz"

    print(f"\n[INFO] Loading sample features from: {sample_path}")
    # Expect a .npz that stores either:
    #   - key "x" (tensor input), or
    #   - multiple keys for dict input.
    data = np.load(sample_path, allow_pickle=True)
    if "x" in data.files:
        sample_features = data["x"].astype(np.float32)
    else:
        # dict input
        sample_features = {k: tf.constant(data[k]) for k in data.files}
        

    # ===== 2.5) Empirical Lipschitz estimate (random perturbations) =====
    # 使用同一批 sample_features 来估计经验 Lipschitz（输出是 utilities 的变化率）
    # 注意：这是经验估计，不是严格上界；通常应 <= 你算出来的 SDP 上界（但也可能因数值/统计抖动而接近）
    x_for_emp = sample_features  # tensor 或 dict 都支持
    emp_L = empirical_lipschitz_estimate(
        model=model,
        x_batch=x_for_emp,
        n_samples=1000,   # 你可以改大，比如 1000；越大越慢
        epsilon=0.01,
        seed=42,
    )
    print("\n========== Empirical Lipschitz Estimate ==========")
    print(f"[RESULT] Empirical Lipschitz (epsilon=0.01, n_samples=200): ~ {emp_L:.6g}")


    stats = verify_decision_robustness(
        model=model, 
        sample_features=sample_features, 
        L_decision=L_decision, # 使用上一步 calculate_decision_lipschitz 的返回值
        lambda_cost=0.5,
        epsilon=0.1
    )

# ==============================================================================
# Main Execution Flow
# ==============================================================================

def main():
    # --------------------------------------------------------------------------
    # 0. 配置参数 (根据你的实际实验设置修改)
    # --------------------------------------------------------------------------
    # 模型路径
    MODEL_PATH = "../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40"
    # 样本数据路径 (.npz)
    DATA_PATH = "sample_features_ECLIFT.npz" 
    
    # 鲁棒性验证参数
    EPSILON = 0.1          # 攻击半径 (L2 Norm)
    LAMBDA_COST = 0.5      # 效用公式权重: Paid - lambda * Cost
    TARGETS = ['paid', 'cost'] 
    TREATMENTS = [0, 1]
    
    # SDP 配置
    SDP_SOLVER = "SCS"     # 或 "MOSEK" (如果安装了)
    P_STRUCTURE = "diag"   # "diag" 速度快且对大网络通常足够紧

    print(f"[配置] Model: {MODEL_PATH}")
    print(f"[配置] Data: {DATA_PATH}")
    print(f"[配置] Epsilon: {EPSILON}, Lambda: {LAMBDA_COST}")

    # --------------------------------------------------------------------------
    # 1. 加载模型
    # --------------------------------------------------------------------------
    print("\n[Step 1] Loading Model...")
    try:
        # compile=False 加快加载速度，且我们不需要训练
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.summary()
    except OSError as e:
        print(f"[Error] 无法加载模型: {e}")
        return

    # --------------------------------------------------------------------------
    # 2. 加载数据样本
    # --------------------------------------------------------------------------
    print("\n[Step 2] Loading Sample Features...")
    if not os.path.exists(DATA_PATH):
        print(f"[Error] 数据文件不存在: {DATA_PATH}")
        # 创建假数据用于测试流程 (如果文件不存在)
        print("[Info] 生成随机假数据用于测试流程...")
        # 注意：这里需要根据你的实际 Input Layer 名字来构造
        sample_features = {
            "dense_input": tf.random.normal((100, 10)), 
            "sparse_input": tf.random.uniform((100, 5), maxval=100, dtype=tf.int32)
        }
    else:
        # data = np.load(DATA_PATH, allow_pickle=True)
        # # 处理 .npz 中可能的存储格式
        # if "x" in data.files:
        #     # 单输入 Tensor
        #     sample_features = tf.constant(data["x"], dtype=tf.float32)
        # else:
        #     # 多输入 Dict
        #     sample_features = {}
        #     for k in data.files:
        #         # 过滤掉标签数据 (如 'y', 'label')，只保留特征
        #         if k not in ['y', 'label', 'target']:
        #             val = data[k]
        #             # 转换类型：浮点转 float32，整型转 int32
        #             if np.issubdtype(val.dtype, np.floating):
        #                 sample_features[k] = tf.constant(val, dtype=tf.float32)
        #             else:
        #                 sample_features[k] = tf.constant(val, dtype=tf.int32)
        data = np.load(DATA_PATH, allow_pickle=True)
        if "x" in data.files:
            sample_features = data["x"].astype(np.float32)
        else:
            # dict input
            sample_features = {k: tf.constant(data[k]) for k in data.files}
    
    print(f"[Info] 样本加载完成。Batch Size: {len(next(iter(sample_features.values()))) if isinstance(sample_features, dict) else len(sample_features)}")

    # --------------------------------------------------------------------------
    # 3. 计算 Lipschitz 上界 (SDP 方法 - Multi-tower 修正版)
    # --------------------------------------------------------------------------
    print("\n[Step 3] Calculating Theoretical Lipschitz Bound (SDP)...")
    
    # 计算各组件 L
    L_shared, tower_map = RobustnessSDP.analyze_multitower_lipschitz(
        model, 
        treatment_order=TREATMENTS, 
        targets=TARGETS
    )
    
    # 计算决策总 L
    L_decision_theoretical = RobustnessSDP.calculate_decision_lipschitz(
        tower_map, 
        targets=TARGETS, 
        lambda_cost=LAMBDA_COST
    )
    
    print(f"\n>>> [Result] User Tower (Shared) L : {L_shared:.6f}")
    print(f">>> [Result] Decision Function Total L : {L_decision_theoretical:.6f}")

    # --------------------------------------------------------------------------
    # 3.5 (可选) 经验 Lipschitz 估计 (Empirical Estimate)
    # --------------------------------------------------------------------------
    # 用来对比 SDP 算出来的上界是否过松 (Over-conservative)
    print("\n[Step 3.5] Estimating Empirical Lipschitz (Random Perturbation)...")
    emp_L = empirical_lipschitz_estimate(
        model=model,
        x_batch=sample_features,
        n_samples=200,    # 采样次数
        epsilon=0.01,     # 微小扰动
        seed=42
    )
    print(f">>> [Result] Empirical L (Approx)      : {emp_L:.6f}")
    
    if emp_L > 0:
        ratio = L_decision_theoretical / emp_L
        print(f"[Info] 上界松弛度 (SDP / Empirical) : {ratio:.2f}x")
        if ratio > 100:
            print("[Warn] SDP 上界可能过于保守，或网络权重极大。")
    
    # --------------------------------------------------------------------------
    # 4. 验证决策鲁棒性
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Decision Robustness...")
    
    # 使用计算出的 SDP 上界进行验证
    # 如果你想测试 empirical L 的效果，可以临时把下面的 L_decision 换成 emp_L 试试
    robustness_stats = verify_decision_robustness(
        model=model,
        sample_features=sample_features,
        L_decision=L_decision_theoretical, 
        lambda_cost=LAMBDA_COST,
        epsilon=EPSILON
    )

    # --------------------------------------------------------------------------
    # 5. 保存报告
    # --------------------------------------------------------------------------
    report = {
        "model_path": MODEL_PATH,
        "config": {
            "epsilon": EPSILON,
            "lambda_cost": LAMBDA_COST,
            "targets": TARGETS
        },
        "lipschitz_metrics": {
            "L_shared": L_shared,
            "L_decision_theoretical": L_decision_theoretical,
            "L_empirical": emp_L,
            "tower_details": tower_map
        },
        "robustness_metrics": {
            "robust_ratio": robustness_stats["robust_ratio"],
            "mean_safe_radius": robustness_stats["mean_safe_radius"]
        }
    }

    # JSON 序列化辅助函数
    def _json_default(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        return str(obj)

    out_file = "robustness_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default, ensure_ascii=False)
    
    print(f"\n[Done] 完整报告已保存至: {out_file}")

if __name__ == "__main__":
    # 设置 Tensorflow 日志等级，减少干扰
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # 禁用 GPU (SDP 求解通常在 CPU，且避免显存冲突)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    main()
