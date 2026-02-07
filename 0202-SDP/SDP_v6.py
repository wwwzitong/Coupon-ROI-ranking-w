import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import cvxpy as cp
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union

# ==============================================================================
# Solver helper
# ==============================================================================
def _try_solve(prob: cp.Problem, prefer: Optional[str] = None, verbose: bool = False) -> None:
    solvers = []
    if prefer:
        solvers.append(prefer)
    solvers += ["MOSEK", "GUROBI", "CVXOPT", "SCS", "ECOS"]

    last_err = None
    for s in solvers:
        try:
            prob.solve(solver=getattr(cp, s), verbose=verbose)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                print(f"solver: {s}")
                return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"SDP solve failed. Last error: {last_err}, status={prob.status}")


# ==============================================================================
# Empirical Lipschitz estimate (unchanged from your code)
# ==============================================================================
def _as_numpy_features(x: Any) -> Any:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out[k] = v.numpy() if tf.is_tensor(v) else np.asarray(v)
        return out
    else:
        return x.numpy() if tf.is_tensor(x) else np.asarray(x)

def _slice_one(x_batch_np: Any, idx: int) -> Any:
    if isinstance(x_batch_np, dict):
        return {k: v[idx:idx+1] for k, v in x_batch_np.items()}
    else:
        return x_batch_np[idx:idx+1]

def _l2_norm_features(delta: Any) -> float:
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
            return np.zeros_like(arr)
        return rng.standard_normal(size=arr.shape).astype(arr.dtype)

def _scale_delta_to_epsilon(delta: Any, epsilon: float) -> Any:
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
    if isinstance(x1, dict):
        out = {}
        for k in x1.keys():
            xv = np.asarray(x1[k])
            dv = np.asarray(delta[k])
            if np.issubdtype(xv.dtype, np.floating):
                out[k] = (xv + dv).astype(xv.dtype)
            else:
                out[k] = xv
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
    x_batch_np = _as_numpy_features(x_batch)
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

        delta = _randn_like_features(x1, rng)
        delta = _scale_delta_to_epsilon(delta, float(epsilon))
        x2 = _add_delta(x1, delta)

        pred1 = model(x1, training=False)
        pred2 = model(x2, training=False)

        u1, _ = RobustnessSDP.compute_utilities_from_predictions(pred1)
        u2, _ = RobustnessSDP.compute_utilities_from_predictions(pred2)

        output_diff = float(np.linalg.norm((u1 - u2).reshape(-1).astype(np.float64)))
        input_diff = float(_l2_norm_features(delta))

        if input_diff > 0:
            ratio = output_diff / input_diff
            max_ratio = max(max_ratio, ratio)

    return float(max_ratio)


# ==============================================================================
# LipSDP (Theorem 2) implementation: LipSDP-Layer variant
# ==============================================================================
class RobustnessSDP:
    """
    LipSDP-Layer (whole-network SDP) for fully-connected feed-forward networks with
    slope-restricted activations (ReLU/LeakyReLU).
    """

    # -----------------------------
    # Utilities computation (your original)
    # -----------------------------
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
        if not isinstance(preds, dict):
            raise ValueError("Model output must be a dict for compute_utilities_from_predictions().")

        k_p0 = f"{paid_prefix}0"
        k_p1 = f"{paid_prefix}1"
        k_c0 = f"{cost_prefix}0"
        k_c1 = f"{cost_prefix}1"

        missing = [k for k in [k_p0, k_p1, k_c0, k_c1] if k not in preds]
        if missing:
            raise ValueError("Missing required prediction keys: " + ", ".join(missing))

        paid0 = RobustnessSDP._to_numpy(preds[k_p0]).reshape(-1)
        paid1 = RobustnessSDP._to_numpy(preds[k_p1]).reshape(-1)
        cost0 = RobustnessSDP._to_numpy(preds[k_c0]).reshape(-1)
        cost1 = RobustnessSDP._to_numpy(preds[k_c1]).reshape(-1)

        u1 = (paid1 - paid0) - float(lambda_cost) * (cost1 - cost0)
        u0 = np.zeros_like(u1)

        U = np.stack([u0, u1], axis=1)
        return U, [0, 1]

    # -----------------------------
    # Fold BN (inference) into affine
    # -----------------------------
    @staticmethod
    def _fold_batchnorm_into_affine(
        W: np.ndarray, b: np.ndarray, bn: tf.keras.layers.BatchNormalization
    ) -> Tuple[np.ndarray, np.ndarray]:
        mv = bn.moving_variance.numpy()
        mm = bn.moving_mean.numpy()
        eps = bn.epsilon

        gamma = bn.gamma.numpy() if bn.scale else np.ones_like(mv)
        beta = bn.beta.numpy() if bn.center else np.zeros_like(mv)

        s = gamma / np.sqrt(mv + eps)          # per-output scaling
        W_new = s[:, None] * W                 # scale rows
        b_new = s * (b - mm) + beta
        return W_new, b_new

    # -----------------------------
    # Extract W_k and sector bounds
    # -----------------------------
    @staticmethod
    def _extract_ffn_layers_for_lipsdp(
        model_seq: tf.keras.Sequential
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """
        Returns:
          W_list: [W0, W1, ..., W_L] where last is output linear (no activation after).
                  Each Wk is shape (n_{k+1}, n_k)
          sector_list: [(alpha1,beta1), ..., (alphaL,betaL)] for hidden activations
                       ReLU -> (0,1), LeakyReLU(a) -> (a,1)
        """
        W_list: List[np.ndarray] = []
        sector_list: List[Tuple[float, float]] = []

        layers = list(model_seq.layers)
        i = 0
        while i < len(layers):
            layer = layers[i]

            if isinstance(layer, (tf.keras.layers.InputLayer, tf.keras.layers.Flatten, tf.keras.layers.Reshape)):
                i += 1
                continue
            if isinstance(layer, tf.keras.layers.Dropout):
                i += 1
                continue

            if isinstance(layer, tf.keras.layers.Dense):
                kernel = layer.kernel.numpy()
                W = kernel.T.astype(np.float64)  # (n_out, n_in)
                b = layer.bias.numpy().astype(np.float64) if (layer.use_bias and layer.bias is not None) \
                    else np.zeros((W.shape[0],), dtype=np.float64)

                # fold BN if immediate next
                if i + 1 < len(layers) and isinstance(layers[i+1], tf.keras.layers.BatchNormalization):
                    W, b = RobustnessSDP._fold_batchnorm_into_affine(W, b, layers[i+1])
                    i += 1

                W_list.append(W)

                # activation inside Dense?
                alpha_beta = None
                act_name = getattr(layer.activation, "__name__", "linear").lower()
                if act_name not in ("linear",):
                    if "relu" in act_name:
                        alpha_beta = (0.0, 1.0)
                    else:
                        raise ValueError(f"Unsupported Dense activation for LipSDP: {act_name}")

                # separate activation layer?
                if alpha_beta is None and i + 1 < len(layers):
                    nxt = layers[i+1]
                    if isinstance(nxt, tf.keras.layers.ReLU):
                        alpha_beta = (0.0, 1.0)
                        i += 1
                    elif isinstance(nxt, tf.keras.layers.LeakyReLU):
                        alpha_beta = (float(nxt.alpha), 1.0)
                        i += 1
                    elif isinstance(nxt, tf.keras.layers.Activation):
                        nm = str(nxt.activation.__name__).lower()
                        if "relu" in nm:
                            alpha_beta = (0.0, 1.0)
                            i += 1
                        else:
                            raise ValueError(f"Unsupported Activation for LipSDP: {nm}")

                if alpha_beta is not None:
                    sector_list.append(alpha_beta)

                i += 1
                continue

            if isinstance(layer, tf.keras.layers.BatchNormalization):
                raise ValueError("BatchNorm must appear right after Dense to be folded (in this implementation).")

            raise ValueError(f"Unsupported layer in LipSDP extractor: {type(layer).__name__}")

        # Need last layer linear output => len(W_list) = len(sector_list) + 1
        # if len(W_list) != len(sector_list) + 1:
        #     raise ValueError(
        #         f"Expected last layer linear. Got len(W_list)={len(W_list)}, len(sector_list)={len(sector_list)}. "
        #         f"Ensure the final Dense has no activation and there is no activation layer after it."
        #     )

        # --- Post-fix for "nonlinear output" ---
        # Classic LipSDP (Theorem 2) implementation expects last layer linear:
        # len(W_list) = len(sector_list) + 1.
        #
        # If your model ends with an activation, you'll get len(W_list) == len(sector_list).
        # In that case we append an identity linear layer as the final output layer.
        if len(W_list) == len(sector_list):
            # Append W_out = I (no extra activation)
            n_last = int(W_list[-1].shape[0])  # dimension of x_L (post-activation output)
            W_list.append(np.eye(n_last, dtype=np.float64))
        elif len(W_list) != len(sector_list) + 1:
            raise ValueError(
                f"Expected len(W_list)=len(sector_list)+1 (linear output). "
                f"Got len(W_list)={len(W_list)}, len(sector_list)={len(sector_list)}. "
                f"Check if there are unsupported layers/activations after the final Dense."
            )

        return W_list, sector_list

    # -----------------------------
    # Build A,B (Eq. 13) and solve Theorem 2 SDP (Eq. 14)
    # -----------------------------
    @staticmethod
    def lipsdp_layer_bound(
        W_list: List[np.ndarray],
        sector_list: List[Tuple[float, float]],
        solver: Optional[str] = None,
        verbose: bool = False
    ) -> float:
        """
        LipSDP-Layer: T = blkdiag(lambda_1 I_{n1}, ..., lambda_l I_{nl}), lambda_k >= 0
        Solve: minimize rho s.t. M(rho, T) << 0
        Return: L = sqrt(rho)
        """
        l = len(sector_list)  # number of hidden activations/layers with nonlinearity
        if l <= 0:
            # purely linear: Lipschitz is spectral norm of overall linear map
            W = W_list[-1]
            for k in range(len(W_list)-2, -1, -1):
                W = W @ W_list[k]
            return float(np.linalg.norm(W, 2))

        n0 = int(W_list[0].shape[1])
        hidden_dims = [int(W.shape[0]) for W in W_list[:-1]]     # n1..nl
        nl = hidden_dims[-1]
        n_hidden_total = int(sum(hidden_dims))
        N = int(n0 + n_hidden_total)  # x = [x0; x1; ...; xl]

        # offsets for x blocks in concatenated x
        x_offsets = [0]              # x0 offset
        cur = n0
        for nk in hidden_dims:
            x_offsets.append(cur)    # xk offset for k>=1
            cur += nk

        # Build A,B as numeric constants (Eq.13 style)
        A = np.zeros((n_hidden_total, N), dtype=np.float64)
        B = np.zeros((n_hidden_total, N), dtype=np.float64)

        row = 0
        for k in range(l):
            Wk = W_list[k].astype(np.float64)         # (n_{k+1}, n_k)
            nk1, nk = Wk.shape
            # A has Wk on block mapping x_k -> z_{k+1}
            A[row:row+nk1, x_offsets[k]:x_offsets[k]+nk] = Wk
            # B selects x_{k+1}
            B[row:row+nk1, x_offsets[k+1]:x_offsets[k+1]+nk1] = np.eye(nk1)
            row += nk1

        # Output linear map uses last hidden x_l
        Wout = W_list[-1].astype(np.float64)          # (n_out, n_l)
        WtW = Wout.T @ Wout                           # (n_l, n_l)

        # Sector: allow mixed by taking conservative global [min alpha, max beta]
        alpha = float(min(a for a, b in sector_list))
        beta = float(max(b for a, b in sector_list))

        rho = cp.Variable(nonneg=True)
        lambdas = cp.Variable(l, nonneg=True)

        # T = blkdiag(lambda_k I_{n_k})
        # T_blocks = [lambdas[k] * np.eye(hidden_dims[k]) for k in range(l)]
        # T = cp.bmat([
        #     [T_blocks[i] if i == j else np.zeros((hidden_dims[i], hidden_dims[j]))
        #      for j in range(l)]
        #     for i in range(l)
        # ])
        t_diag = cp.hstack([lambdas[k] * np.ones(hidden_dims[k]) for k in range(l)])
        T = cp.diag(t_diag)


        Q11 = (-2.0 * alpha * beta) * T
        Q12 = (alpha + beta) * T
        Q22 = (-2.0) * T
        Q = cp.bmat([[Q11, Q12],
                     [Q12, Q22]])

        AB = np.vstack([A, B])                     # (2n, N)
        ABc = cp.Constant(AB)

        # Drho = diag(-rho I_{n0}, 0, ..., 0, Wout^T Wout) as in Eq.(14)
        Drho = np.zeros((N, N), dtype=np.float64)
        # put WtW on x_l block
        xL_off = x_offsets[l]
        Drho[xL_off:xL_off+nl, xL_off:xL_off+nl] = WtW
        Drhoc = cp.Constant(Drho)

        # -rho I on x0 block
        E0 = np.zeros((n0, N), dtype=np.float64)
        E0[:, 0:n0] = np.eye(n0)
        E0c = cp.Constant(E0)

        M = ABc.T @ Q @ ABc + Drhoc - rho * (E0c.T @ E0c)

        constraints = [M << 0]
        prob = cp.Problem(cp.Minimize(rho), constraints)
        _try_solve(prob, prefer=solver, verbose=verbose)

        if rho.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"LipSDP failed: status={prob.status}")

        return float(np.sqrt(max(float(rho.value), 0.0)))

    @staticmethod
    def get_sequential_lipschitz_lipsdp(
        model_seq: tf.keras.Sequential,
        solver: Optional[str] = None,
        verbose: bool = False
    ) -> float:
        W_list, sector_list = RobustnessSDP._extract_ffn_layers_for_lipsdp(model_seq)
        return RobustnessSDP.lipsdp_layer_bound(W_list, sector_list, solver=solver, verbose=verbose)

    # -----------------------------
    # Multi-tower analysis using whole-path LipSDP (NO product)
    # -----------------------------
    @staticmethod
    def analyze_multitower_lipschitz_lipsdp(
        model,
        treatment_order,
        targets,
        solver: Optional[str] = None,
        verbose: bool = False
    ):
        print("\n[LipSDP] Computing shared tower Lipschitz (optional, for reporting)...")
        try:
            user_tower = model.user_tower
        except AttributeError:
            print("  [Warn] model.user_tower missing. Trying get_layer('user_tower')...")
            user_tower = model.get_layer("user_tower")

        L_shared = RobustnessSDP.get_sequential_lipschitz_lipsdp(user_tower, solver=solver, verbose=verbose)
        print(f"  => L_shared (LipSDP, shared only) = {L_shared:.6f}")

        # Build all task tower names
        task_names = []
        for target in targets:
            for treatment in treatment_order:
                task_names.append(f"{target}_treatment_{treatment}_tower")

        tower_lipschitz_map: Dict[str, float] = {}

        print("\n[LipSDP] Computing FULL-PATH Lipschitz for each (shared + head) with ONE SDP...")
        for name in task_names:
            try:
                if hasattr(model, "task_towers") and name in model.task_towers:
                    head = model.task_towers[name]
                else:
                    head = model.get_layer(name)

                # Full path sequential: shared layers + head layers
                full_path = tf.keras.Sequential(list(user_tower.layers) + list(head.layers))

                print(f"  - Solving LipSDP for full path: {name} ...")
                L_path = RobustnessSDP.get_sequential_lipschitz_lipsdp(full_path, solver=solver, verbose=verbose)
                tower_lipschitz_map[name] = L_path
                print(f"    => L_path (LipSDP, {name}) = {L_path:.6f}")

            except Exception as e:
                print(f"  [Error] Could not analyze path '{name}': {e}")
                tower_lipschitz_map[name] = float("inf")

        return L_shared, tower_lipschitz_map

    @staticmethod
    def calculate_decision_lipschitz(tower_lipschitz_map, targets, lambda_cost=0.5):
        p1 = tower_lipschitz_map.get(f"{targets[0]}_treatment_1_tower", 0.0)
        p0 = tower_lipschitz_map.get(f"{targets[0]}_treatment_0_tower", 0.0)
        c1 = tower_lipschitz_map.get(f"{targets[1]}_treatment_1_tower", 0.0)
        c0 = tower_lipschitz_map.get(f"{targets[1]}_treatment_0_tower", 0.0)
        return p1 + p0 + abs(lambda_cost) * c1 + abs(lambda_cost) * c0


# ==============================================================================
# Decision robustness verification (your original, unchanged)
# ==============================================================================
def verify_decision_robustness(model, sample_features, L_decision, lambda_cost=0.5, epsilon=0.1):
    print("\n========== 4. 决策鲁棒性验证 (Decision Robustness) ==========")
    print(f"[配置] Epsilon (扰动半径): {epsilon}")
    print(f"[配置] Lambda Cost: {lambda_cost}")
    print(f"[参数] L_decision (Diff Lipschitz): {L_decision:.6f}")

    print("正在进行模型推理...")
    preds = model(sample_features, training=False)

    def to_np(x):
        return x.numpy().flatten() if tf.is_tensor(x) else np.asarray(x).flatten()

    try:
        p0 = to_np(preds["paid_treatment_0_tower"])
        p1 = to_np(preds["paid_treatment_1_tower"])
        c0 = to_np(preds["cost_treatment_0_tower"])
        c1 = to_np(preds["cost_treatment_1_tower"])
    except KeyError:
        p0 = to_np(preds.get("paid_treatment_0", np.zeros(1)))
        p1 = to_np(preds.get("paid_treatment_1", np.zeros(1)))
        c0 = to_np(preds.get("cost_treatment_0", np.zeros(1)))
        c1 = to_np(preds.get("cost_treatment_1", np.zeros(1)))
        print("[Warn] 使用了备用 Key (不带 _tower 后缀)")

    u0 = p0 - lambda_cost * c0
    u1 = p1 - lambda_cost * c1
    diff_u = u1 - u0

    margin = np.abs(diff_u)
    original_decision = (diff_u > 0).astype(int)

    valid_L = max(L_decision, 1e-8)
    safe_radius = margin / valid_L
    is_robust = safe_radius >= epsilon

    n_samples = len(margin)
    robust_count = np.sum(is_robust)
    robust_ratio = robust_count / max(n_samples, 1)

    print("\n[验证结果报告]")
    print(f"样本总数          : {n_samples}")
    print(f"鲁棒样本数        : {robust_count}")
    print(f"鲁棒比例 (Robust%) : {robust_ratio * 100:.2f}%")
    print(f"-" * 30)
    print(f"平均 Margin       : {np.mean(margin):.6f}")
    print(f"平均 Safe Radius  : {np.mean(safe_radius):.6f}")
    print(f"最小 Safe Radius  : {np.min(safe_radius):.6f}")

    return {
        "robust_ratio": float(robust_ratio),
        "mean_safe_radius": float(np.mean(safe_radius)),
        "is_robust_indices": is_robust,
        "margins": margin,
        "original_decision": original_decision,
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    MODEL_PATH = "../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40"
    DATA_PATH = "sample_features_ECLIFT.npz"

    EPSILON = 0.1
    LAMBDA_COST = 0.5
    TARGETS = ["paid", "cost"]
    TREATMENTS = [0, 1]

    SDP_SOLVER = "MOSEK"   # if available; otherwise will fall back

    print(f"[配置] Model: {MODEL_PATH}")
    print(f"[配置] Data : {DATA_PATH}")
    print(f"[配置] Epsilon: {EPSILON}, Lambda: {LAMBDA_COST}")
    print(f"[配置] SDP Solver preference: {SDP_SOLVER}")

    # 1) Load model
    print("\n[Step 1] Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.summary()

    # 2) Load sample features
    print("\n[Step 2] Loading Sample Features...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=True)
    if "x" in data.files:
        sample_features = data["x"].astype(np.float32)
    else:
        sample_features = {k: tf.constant(data[k]) for k in data.files}

    bs = len(next(iter(sample_features.values()))) if isinstance(sample_features, dict) else len(sample_features)
    print(f"[Info] Batch size: {bs}")

    # 3) LipSDP full-path bounds (NO product)
    print("\n[Step 3] Computing Lipschitz bounds via LipSDP (whole-path SDP)...")
    L_shared, tower_map = RobustnessSDP.analyze_multitower_lipschitz_lipsdp(
        model=model,
        treatment_order=TREATMENTS,
        targets=TARGETS,
        solver=SDP_SOLVER,
        verbose=False
    )

    L_decision_theoretical = RobustnessSDP.calculate_decision_lipschitz(
        tower_lipschitz_map=tower_map,
        targets=TARGETS,
        lambda_cost=LAMBDA_COST
    )

    print(f"\n>>> [Result] Shared-only L (LipSDP)        : {L_shared:.6f}")
    print(f">>> [Result] Decision function L_dec (sum): {L_decision_theoretical:.6f}")

    # 3.5) Empirical estimate (optional)
    print("\n[Step 3.5] Empirical Lipschitz (random perturbations)...")
    emp_L = empirical_lipschitz_estimate(
        model=model,
        x_batch=sample_features,
        n_samples=200,
        epsilon=0.01,
        seed=42
    )
    print(f">>> [Result] Empirical L (approx): {emp_L:.6f}")
    if emp_L > 0 and np.isfinite(L_decision_theoretical):
        print(f"[Info] Upper-bound looseness (LipSDP / Emp): {L_decision_theoretical / emp_L:.2f}x")

    # 4) Decision robustness
    print("\n[Step 4] Verifying decision robustness...")
    stats = verify_decision_robustness(
        model=model,
        sample_features=sample_features,
        L_decision=L_decision_theoretical,
        lambda_cost=LAMBDA_COST,
        epsilon=EPSILON
    )

    # 5) Save report
    report = {
        "model_path": MODEL_PATH,
        "config": {
            "epsilon": EPSILON,
            "lambda_cost": LAMBDA_COST,
            "targets": TARGETS,
            "solver_prefer": SDP_SOLVER
        },
        "lipschitz_metrics": {
            "L_shared": L_shared,
            "tower_details_fullpath": tower_map,
            "L_decision_theoretical": L_decision_theoretical,
            "L_empirical": emp_L
        },
        "robustness_metrics": {
            "robust_ratio": stats["robust_ratio"],
            "mean_safe_radius": stats["mean_safe_radius"]
        }
    }

    def _json_default(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        return str(obj)

    out_file = "robustness_report_lipsdp.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default, ensure_ascii=False)
    print(f"\n[Done] Report saved to: {out_file}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
