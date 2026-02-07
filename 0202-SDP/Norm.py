import os
import json
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------
# 1. 基础辅助函数（处理数据转换、扰动、范数计算）
# ---------------------------------------------------------

def _as_numpy_features(x: Any) -> Any:
    """将 Tensor 或 Dict[Tensor] 转换为 Numpy 格式"""
    if isinstance(x, dict):
        return {k: (v.numpy() if tf.is_tensor(v) else np.asarray(v)) for k, v in x.items()}
    return x.numpy() if tf.is_tensor(x) else np.asarray(x)

def _slice_one(x_batch_np: Any, idx: int) -> Any:
    """从 Batch 数据中切分出第 idx 个样本"""
    if isinstance(x_batch_np, dict):
        return {k: v[idx:idx+1] for k, v in x_batch_np.items()}
    return x_batch_np[idx:idx+1]

def _l2_norm_features(delta: Any) -> float:
    """计算 L2 范数（支持 Dict 结构）"""
    if isinstance(delta, dict):
        s = 0.0
        for v in delta.values():
            vv = np.asarray(v).astype(np.float64)
            s += float(np.sum(vv.ravel() ** 2))
        return float(np.sqrt(s))
    return float(np.linalg.norm(np.asarray(delta).reshape(-1).astype(np.float64)))

def _randn_like_features(x1: Any, rng: np.random.Generator) -> Any:
    """生成与输入结构一致的随机噪声"""
    if isinstance(x1, dict):
        delta = {}
        for k, v in x1.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.floating):
                delta[k] = rng.standard_normal(size=arr.shape).astype(arr.dtype)
            else:
                delta[k] = np.zeros_like(arr)
        return delta
    arr = np.asarray(x1)
    if not np.issubdtype(arr.dtype, np.floating):
        return np.zeros_like(arr)
    return rng.standard_normal(size=arr.shape).astype(arr.dtype)

def _scale_delta_to_epsilon(delta: Any, epsilon: float) -> Any:
    """缩放噪声向量使其模长等于 epsilon"""
    norm = _l2_norm_features(delta)
    if norm <= 0.0: return delta
    scale = float(epsilon) / norm
    if isinstance(delta, dict):
        return {k: (np.asarray(v) * scale).astype(np.asarray(v).dtype) for k, v in delta.items()}
    return (np.asarray(delta) * scale).astype(np.asarray(delta).dtype)

def _add_delta(x1: Any, delta: Any) -> Any:
    """执行 x1 + delta"""
    if isinstance(x1, dict):
        return {k: (np.asarray(x1[k]) + np.asarray(delta[k])).astype(np.asarray(x1[k]).dtype) 
                if np.issubdtype(np.asarray(x1[k]).dtype, np.floating) else x1[k] for k in x1}
    return (np.asarray(x1) + np.asarray(delta)).astype(np.asarray(x1).dtype)

# ---------------------------------------------------------
# 2. 核心计算类：基于范数的 Lipschitz 分析
# ---------------------------------------------------------

class RobustnessNormBased:
    """
    针对 Multi-tower 结构的 Lipschitz 验证工具
    使用直接的矩阵谱范数 (||W||2) 代替 SDP
    """

    @staticmethod
    def get_layer_lipschitz(layer: tf.keras.layers.Layer) -> float:
        """
        计算单层的 Lipschitz 常数
        """
        # --- Dense 层 ---
        if isinstance(layer, tf.keras.layers.Dense):
            W = layer.kernel.numpy()  # (n_in, n_out)
            # 线性部分的 L = 矩阵的谱范数（最大奇异值）
            L_lin = float(np.linalg.norm(W, ord=2))
            
            # 激活函数的 L
            act_name = getattr(layer.activation, "__name__", "linear").lower()
            if "relu" in act_name or "tanh" in act_name or "linear" in act_name:
                L_act = 1.0
            elif "sigmoid" in act_name:
                L_act = 0.25
            else:
                L_act = 1.0  # 保守估计
            return L_lin * L_act

        # --- BatchNorm 层 ---
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            mv = layer.moving_variance.numpy()
            eps = layer.epsilon
            gamma = layer.gamma.numpy() if layer.scale else np.ones_like(mv)
            # BN 在推理时是逐元素缩放，Lipschitz 为缩放因子的最大绝对值
            scale = np.abs(gamma / np.sqrt(mv + eps))
            return float(np.max(scale))

        # --- 其他层 ---
        return 1.0  # 如 Dropout (推理时为 Identity), Flatten, Reshape

    @staticmethod
    def get_sequential_lipschitz(model_seq: tf.keras.Sequential) -> float:
        """计算串行模块的总 Lipschitz 常数 (L_total = L1 * L2 * ...)"""
        L_total = 1.0
        for layer in model_seq.layers:
            L_layer = RobustnessNormBased.get_layer_lipschitz(layer)
            if L_layer != 1.0:
                print(f"    - Layer {layer.name:<20} | L_layer: {L_layer:.4f}")
            L_total *= L_layer
        return L_total

    @staticmethod
    def analyze_multitower_lipschitz_old(model, treatment_order, targets):
        """解析 Multi-tower 模型结构并计算各路径 L"""
        print("\n[Analysis] Calculating Lipschitz Bounds using Spectral Norms...")
        
        # 1. Shared Bottom (User Tower)
        try:
            user_tower = model.user_tower
        except AttributeError:
            user_tower = model.get_layer('user_tower')

        L_shared = RobustnessNormBased.get_sequential_lipschitz(user_tower)
        print(f"  => L_shared (User Tower) = {L_shared:.6f}")

        # 2. Heads (Task Towers)
        tower_map = {}
        for target in targets:
            for treatment in treatment_order:
                name = f"{target}_treatment_{treatment}"
                try:
                    tower = model.task_towers[name] if hasattr(model, 'task_towers') else model.get_layer(name)
                    L_head = RobustnessNormBased.get_sequential_lipschitz(tower)
                    L_path = L_shared * L_head
                    tower_map[name] = L_path
                    print(f"  => L_path ({name:<30}) = {L_path:.6f}")
                except Exception as e:
                    print(f"  [Warn] Skipping {name}: {e}")
                    tower_map[name] = 0.0
        
        return L_shared, tower_map

    @staticmethod
    def analyze_multitower_lipschitz(model, treatment_order, targets):
        """解析 Multi-tower 模型结构并计算各路径 L"""
        print("\n[Analysis] Calculating Lipschitz Bounds using Spectral Norms...")
        
        # 打印模型所有层名，方便你核对
        all_layer_names = [l.name for l in model.layers]
        print(f"[Debug] Model layers found: {all_layer_names}")

        # 1. Shared Bottom (User Tower)
        # 尝试常见的命名
        user_tower = None
        for name in ['user_tower', 'shared_bottom', 'user_embedding']:
            try:
                user_tower = model.get_layer(name)
                print(f"  [Found] User Tower located as layer: '{name}'")
                break
            except: continue
        
        if user_tower is None:
            # 如果没找到，尝试取模型的第一层（通常是 User Tower）
            user_tower = model.layers[0]
            print(f"  [Warn] User Tower not found by name. Using first layer: '{user_tower.name}'")

        L_shared = RobustnessNormBased.get_sequential_lipschitz(user_tower)
        print(f"  => L_shared = {L_shared:.6f}")

        # 2. Heads (Task Towers)
        tower_map = {}
        for target in targets:
            for treatment in treatment_order:
                # 尝试两种常见的命名后缀
                possible_names = [
                    f"{target}_treatment_{treatment}_tower",
                    f"{target}_treatment_{treatment}",
                    f"tower_{target}_{treatment}"
                ]
                
                tower = None
                found_name = ""
                for name in possible_names:
                    try:
                        # 如果模型有 task_towers 属性
                        if hasattr(model, 'task_towers') and name in model.task_towers:
                            tower = model.task_towers[name]
                        else:
                            tower = model.get_layer(name)
                        found_name = name
                        break
                    except: continue
                
                if tower:
                    print(f"  [Found] Analyzing {found_name}...")
                    L_head = RobustnessNormBased.get_sequential_lipschitz(tower)
                    L_path = L_shared * L_head
                    tower_map[found_name] = L_path
                    # 为了兼容后续逻辑，统一映射回标准的 key
                    tower_map[f"{target}_treatment_{treatment}_tower"] = L_path 
                else:
                    print(f"  [Error] Could not find tower for Target:{target}, Treatment:{treatment}")
                    # 如果实在找不到，根据你的模型结构，这里可以手动指定
                    # tower_map[f"{target}_treatment_{treatment}_tower"] = 1.0 
        
        return L_shared, tower_map

    @staticmethod
    def calculate_decision_lipschitz(tower_map, targets, lambda_cost=0.5):
        """计算最终决策函数 U = (P1-P0) - lambda*(C1-C0) 的 Lipschitz 常数"""
        p1 = tower_map.get(f"{targets[0]}_treatment_1_tower", 0.0)
        p0 = tower_map.get(f"{targets[0]}_treatment_0_tower", 0.0)
        c1 = tower_map.get(f"{targets[1]}_treatment_1_tower", 0.0)
        c0 = tower_map.get(f"{targets[1]}_treatment_0_tower", 0.0)

        # L_sum <= Sum(L_parts)
        L_decision = p1 + p0 + abs(lambda_cost) * (c1 + c0)
        return L_decision

# ---------------------------------------------------------
# 3. 经验估算与验证函数
# ---------------------------------------------------------

def empirical_lipschitz_estimate(model, x_batch, n_samples=500, epsilon=0.01, seed=42):
    """经验估计（随机采样计算最大变化率）"""
    x_batch_np = _as_numpy_features(x_batch)
    B = next(iter(x_batch_np.values())).shape[0] if isinstance(x_batch_np, dict) else x_batch_np.shape[0]
    rng = np.random.default_rng(seed)
    max_ratio = 0.0

    print(f"Running {n_samples} empirical samples...")
    for _ in range(n_samples):
        idx = rng.integers(0, B)
        x1 = _slice_one(x_batch_np, idx)
        delta = _scale_delta_to_epsilon(_randn_like_features(x1, rng), epsilon)
        x2 = _add_delta(x1, delta)

        p1 = model(x1, training=False)
        p2 = model(x2, training=False)

        # 这里假设输出包含 paid_treatment_1 等 Key，计算 Utility 差值
        def get_u(p):
            # 简化版 Utility 计算，仅供参考，应根据你的 compute_utilities_from_predictions 逻辑调整
            u = (p['paid_treatment_1'] - p['paid_treatment_0']) - 0.5 * \
                (p['cost_treatment_1'] - p['cost_treatment_0'])
            return u.numpy().flatten()

        diff_out = np.linalg.norm(get_u(p1) - get_u(p2))
        ratio = diff_out / epsilon
        if ratio > max_ratio: max_ratio = ratio
    return max_ratio

def verify_decision_robustness(model, sample_features, L_decision, lambda_cost=0.5, epsilon=0.1):
    """计算决策鲁棒性统计数据"""
    print("\n========== Decision Robustness Verification ==========")
    preds = model(sample_features, training=False)
    
    def to_np(k): return preds[k].numpy().flatten()
    
    u1 = (to_np('paid_treatment_1') - to_np('paid_treatment_0')) - \
         lambda_cost * (to_np('cost_treatment_1') - to_np('cost_treatment_0'))
    
    margin = np.abs(u1) # 因为 u0 固定为 0
    safe_radius = margin / max(L_decision, 1e-9)
    is_robust = safe_radius >= epsilon

    print(f"Robust Samples: {np.sum(is_robust)} / {len(margin)} ({np.mean(is_robust)*100:.2f}%)")
    print(f"Average Safe Radius: {np.mean(safe_radius):.6f}")
    return is_robust

# ---------------------------------------------------------
# 4. 主执行流
# ---------------------------------------------------------

def main():
    # 配置
    MODEL_PATH = "../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40"
    DATA_PATH = "sample_features_ECLIFT.npz"
    TARGETS = ['paid', 'cost']
    TREATMENTS = [0, 1]
    LAMBDA = 0.5
    EPSILON_TEST = 0.1

    # 1. 加载
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 2. 理论计算 (谱范数法)
    L_shared, tower_map = RobustnessNormBased.analyze_multitower_lipschitz(model, TREATMENTS, TARGETS)
    L_dec_theoretical = RobustnessNormBased.calculate_decision_lipschitz(tower_map, TARGETS, LAMBDA)
    
    print(f"\n>>> Theoretical Lipschitz Bound (L_dec): {L_dec_theoretical:.6f}")

    # 3. 经验验证与鲁棒性评估
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        sample_features = {k: tf.constant(data[k]) for k in data.files}
        
        # 经验估计对比
        # emp_L = empirical_lipschitz_estimate(model, sample_features)
        # print(f">>> Empirical Lipschitz Estimate: {emp_L:.6f}")
        
        # 鲁棒性验证
        verify_decision_robustness(model, sample_features, L_dec_theoretical, LAMBDA, EPSILON_TEST)

if __name__ == "__main__":
    main()