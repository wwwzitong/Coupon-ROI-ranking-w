#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated single-file robustness script for your EcomDFCL_regretNet_rplusc model.

What this script does (all in one file):
1) Loads a saved tf.keras model (compile=False).
2) Computes a *more complete* Lipschitz upper bound for utility u1 under L2 perturbations
   of **dense continuous features only** (consistent with your delta generator that doesn't perturb sparse/id).
   The bound includes:
   - Dense preprocessing: max(x,0), optional log1p, normalization by global std (or clipped fallback)
   - user_tower: Dense spectral norms + activations + BatchNorm (moving stats) + Dropout (training/inference)
   - task_towers: same
   - utility combination: u1 = (paid1-paid0) - lambda*(cost1-cost0)
3) Optionally computes empirical Lipschitz estimate via random perturbations on a batch.
4) Optionally computes decision robustness ratio with the margin rule (same as your previous script).

Notes / Assumptions:
- This bound is for perturbations on dense features only. Sparse/string/id inputs are treated as constant
  (same as your random_delta_like perturbation generator logic in prior script).
- BatchNorm: uses moving stats (inference-style) even if training=True, because training-mode global bound
  is not well-defined without additional assumptions. This matches most robustness verification settings.
- If your model was trained without providing dense_stats (so _dense_global_std is None),
  your preprocessing uses batch std, whose global Lipschitz can be unbounded. We therefore clip std
  to a lower bound `STD_LOWER_BOUND` to obtain a computable (but assumption-dependent) bound.

Usage:
  python integrated_robustness.py \
    --model_path ../final-ECLIFT/model/xxx \
    --sample_npz sample_features_ECLIFT.npz \
    --lambda_cost 0.5 \
    --epsilon 1000 \
    --emp_n 1000 --emp_eps 0.01 \
    --out_json sdp_lipschitz_report.json

The sample .npz:
- if contains key "x": treated as tensor input
- else: treated as dict input with keys matching model inputs
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


# -----------------------------
# Helpers: perturbations (dense-only consistent)
# -----------------------------

def _slice_one_sample(features: Any, idx: int) -> Any:
    """Take sample idx from batch, keep structure (tensor or dict)."""
    if isinstance(features, dict):
        out = {}
        for k, v in features.items():
            v_np = v.numpy() if tf.is_tensor(v) else np.asarray(v)
            out[k] = v_np[idx:idx+1]
        return out
    x_np = features if isinstance(features, np.ndarray) else np.asarray(features)
    return x_np[idx:idx+1]


def _add_delta(features: Any, delta: Any) -> Any:
    """features + delta, keep structure; non-numeric fields are not perturbed."""
    if isinstance(features, dict):
        out = {}
        for k in features.keys():
            v = features[k]
            d = delta.get(k, None) if isinstance(delta, dict) else None
            if d is None:
                out[k] = v
                continue

            v_np = v.numpy() if tf.is_tensor(v) else np.asarray(v)
            if v_np.dtype.kind not in ("f", "i", "u"):
                out[k] = v
                continue
            out[k] = v_np + np.asarray(d, dtype=v_np.dtype)
        return out
    return features + delta


def _random_delta_like_dense_only(x: Any, epsilon: float, dense_keys: Optional[List[str]] = None) -> Any:
    """
    Random L2 perturbation with total norm epsilon.
    - If dict input: only perturb float tensors whose key is in dense_keys (if provided),
      else perturb all float tensors.
    - Non-float / ids / strings: do not perturb.
    """
    if isinstance(x, dict):
        deltas: Dict[str, np.ndarray] = {}
        flats = []

        keys = list(x.keys())
        use_keys = set(dense_keys) if dense_keys else None

        for k in keys:
            v = x[k]
            v_np = v.numpy() if tf.is_tensor(v) else np.asarray(v)

            if use_keys is not None and k not in use_keys:
                d = np.zeros_like(v_np)
            else:
                if v_np.dtype.kind in ("f",):
                    d = np.random.randn(*v_np.shape).astype(v_np.dtype)
                else:
                    d = np.zeros_like(v_np)

            deltas[k] = d
            flats.append(d.reshape(-1))

        flat = np.concatenate(flats, axis=0) if flats else np.zeros((0,), dtype=np.float32)
        norm = np.linalg.norm(flat)
        if norm < 1e-12:
            return deltas

        scale = epsilon / norm
        for k in deltas.keys():
            deltas[k] = deltas[k] * scale
        return deltas

    # tensor input
    x_np = x if isinstance(x, np.ndarray) else np.asarray(x)
    d = np.random.randn(*x_np.shape).astype(x_np.dtype)
    norm = np.linalg.norm(d.reshape(-1))
    if norm < 1e-12:
        return np.zeros_like(x_np)
    return d / norm * epsilon


def _l2_norm_delta(delta: Any) -> float:
    if isinstance(delta, dict):
        flat = np.concatenate([np.asarray(delta[k]).reshape(-1) for k in delta.keys()], axis=0)
        return float(np.linalg.norm(flat))
    return float(np.linalg.norm(np.asarray(delta)))


# -----------------------------
# Utility from predictions (your incremental decision)
# -----------------------------

class Utility:
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
        u1 = (paid1 - paid0) - lambda*(cost1 - cost0)
        U = [u0=0, u1] so argmax(U) matches threshold.
        """
        if not isinstance(preds, dict):
            raise ValueError("Model output must be a dict.")

        k_p0 = f"{paid_prefix}0"
        k_p1 = f"{paid_prefix}1"
        k_c0 = f"{cost_prefix}0"
        k_c1 = f"{cost_prefix}1"

        missing = [k for k in [k_p0, k_p1, k_c0, k_c1] if k not in preds]
        if missing:
            raise ValueError("Missing required prediction keys: " + ", ".join(missing))

        paid0 = Utility._to_numpy(preds[k_p0]).reshape(-1)
        paid1 = Utility._to_numpy(preds[k_p1]).reshape(-1)
        cost0 = Utility._to_numpy(preds[k_c0]).reshape(-1)
        cost1 = Utility._to_numpy(preds[k_c1]).reshape(-1)

        u1 = (paid1 - paid0) - float(lambda_cost) * (cost1 - cost0)
        u0 = np.zeros_like(u1)
        U = np.stack([u0, u1], axis=1)
        return U, [0, 1]


# -----------------------------
# Empirical Lipschitz estimate
# -----------------------------

def empirical_lipschitz_estimate(
    model: tf.keras.Model,
    features_batch: Any,
    lambda_cost: float,
    n_samples: int = 1000,
    epsilon: float = 0.01,
    dense_keys: Optional[List[str]] = None,
) -> float:
    if isinstance(features_batch, dict):
        any_key = next(iter(features_batch.keys()))
        B = (features_batch[any_key].shape[0] if tf.is_tensor(features_batch[any_key])
             else np.asarray(features_batch[any_key]).shape[0])
    else:
        B = len(features_batch)

    max_ratio = 0.0
    for _ in range(int(n_samples)):
        idx = np.random.randint(0, B)
        x = _slice_one_sample(features_batch, idx)
        delta = _random_delta_like_dense_only(x, epsilon=epsilon, dense_keys=dense_keys)

        x1 = x
        x2 = _add_delta(x, delta)

        pred1 = model(x1, training=False)
        pred2 = model(x2, training=False)

        u1, _ = Utility.compute_utilities_from_predictions(pred1, lambda_cost=lambda_cost)
        u2, _ = Utility.compute_utilities_from_predictions(pred2, lambda_cost=lambda_cost)

        out_diff = float(np.linalg.norm(u1 - u2))
        in_diff = _l2_norm_delta(delta)
        if in_diff < 1e-12:
            continue
        max_ratio = max(max_ratio, out_diff / in_diff)

    return float(max_ratio)


# -----------------------------
# Lipschitz bound (more complete) for your architecture
# -----------------------------

def _activation_lip(act: Any) -> float:
    if act is None:
        return 1.0
    name = getattr(act, "__name__", str(act)).lower()
    if "relu" in name:
        return 1.0
    if "tanh" in name:
        return 1.0
    if "sigmoid" in name:
        return 0.25
    if "linear" in name:
        return 1.0
    return 1.0


def _spectral_norm_dense(kernel_np: np.ndarray, n_iter: int = 30) -> float:
    """
    Dense kernel shape: (in, out), forward y = x @ W (+b).
    Return ||W||_2 via power iteration.
    """
    W = np.asarray(kernel_np)
    if W.ndim != 2:
        raise ValueError(f"Dense kernel must be 2D, got {W.shape}")

    out_dim = W.shape[1]
    v = np.ones((out_dim,), dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)

    for _ in range(max(1, int(n_iter))):
        v = W.T @ (W @ v)
        v /= (np.linalg.norm(v) + 1e-12)

    wv = W @ v
    return float(np.linalg.norm(wv) / (np.linalg.norm(v) + 1e-12))


def _batchnorm_lip_inference(bn: tf.keras.layers.BatchNormalization) -> float:
    if not hasattr(bn, "moving_variance") or bn.moving_variance is None:
        return 1.0
    var = bn.moving_variance.numpy()
    eps = float(getattr(bn, "epsilon", 1e-3))
    gamma = getattr(bn, "gamma", None)
    gamma_np = np.ones_like(var, dtype=np.float32) if gamma is None else gamma.numpy()
    scale = gamma_np / np.sqrt(var + eps)
    return float(np.max(np.abs(scale)))


def _dropout_lip(rate: float, training: bool) -> float:
    if not training:
        return 1.0
    keep = 1.0 - float(rate)
    return float(1.0 / max(keep, 1e-12))


def _blockdiag_scale_lip(scales_1d: np.ndarray) -> float:
    s = np.asarray(scales_1d).reshape(-1)
    if s.size == 0:
        return 1.0
    return float(np.max(np.abs(s)))


def _tower_lipschitz(model_or_seq: tf.keras.Model, training: bool, bn_use_moving_stats: bool = True) -> float:
    L = 1.0
    for layer in getattr(model_or_seq, "layers", []):
        if isinstance(layer, tf.keras.layers.Dense):
            L *= _spectral_norm_dense(layer.kernel.numpy())
            L *= _activation_lip(layer.activation)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            L *= _batchnorm_lip_inference(layer) if bn_use_moving_stats else 1.0
        elif isinstance(layer, tf.keras.layers.Dropout):
            L *= _dropout_lip(layer.rate, training=training)
        elif isinstance(layer, tf.keras.layers.Activation):
            L *= _activation_lip(layer.activation)
        else:
            # If you later add Add/Multiply/LayerNorm/Attention, handle explicitly here.
            L *= 1.0
    return float(L)


@dataclass
class LipschitzBoundReport:
    L_pre_dense: float
    L_user_tower: float
    L_heads_logits: Dict[str, float]
    L_utility_u1: float
    lambda_cost: float
    assumptions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assumptions": self.assumptions,
            "L_pre_dense": float(self.L_pre_dense),
            "L_user_tower": float(self.L_user_tower),
            "L_heads_logits": {k: float(v) for k, v in self.L_heads_logits.items()},
            "L_utility_u1": float(self.L_utility_u1),
            "lambda_cost": float(self.lambda_cost),
        }


def compute_model_lipschitz_u1_upper_bound(
    model: tf.keras.Model,
    lambda_cost: float = 0.5,
    training: bool = False,
    std_lower_bound: float = 1e-3,
    dense_feature_names: Optional[List[str]] = None,
    bn_use_moving_stats: bool = True,
) -> LipschitzBoundReport:
    """
    Compute Lipschitz upper bound for u1 w.r.t. dense-feature perturbation.
    This requires:
      - model has attributes: user_tower, task_towers
      - (optional) model has _dense_global_std (Tensor) and dense_feature_names
    If not found, will fall back to provided dense_feature_names/std clipping.
    """
    # ---- dense preprocessing bound ----
    # preprocessing Lipschitz dominated by normalization by std
    std_tensor = getattr(model, "_dense_global_std", None)
    if std_tensor is not None:
        std = std_tensor.numpy().astype(np.float32).reshape(-1)
        std_eff = np.maximum(std, 1e-8)
        dense_std_source = "model._dense_global_std"
        n_dense = std_eff.size
    else:
        # fallback: need number of dense features
        if dense_feature_names is None:
            # last resort: try to use model.dense_feature_names
            dense_feature_names = getattr(model, "dense_feature_names", None)
        if dense_feature_names is None:
            raise ValueError("Cannot infer dense feature count. Provide --dense_keys or ensure model has dense_feature_names.")
        n_dense = len(dense_feature_names)
        std_eff = np.full((n_dense,), float(std_lower_bound), dtype=np.float32)
        dense_std_source = f"clipped_batch_std_lower_bound({std_lower_bound:g})"

    dense_scales = 1.0 / std_eff
    L_pre_dense = _blockdiag_scale_lip(dense_scales)

    # ---- user tower ----
    user_tower = getattr(model, "user_tower", None)
    if user_tower is None:
        raise AttributeError("Model does not have attribute 'user_tower'.")
    L_user = _tower_lipschitz(user_tower, training=training, bn_use_moving_stats=bn_use_moving_stats)

    # ---- heads ----
    task_towers = getattr(model, "task_towers", None)
    if task_towers is None or not isinstance(task_towers, dict):
        raise AttributeError("Model does not have attribute 'task_towers' as a dict.")

    L_heads: Dict[str, float] = {}
    for tower_name, tower in task_towers.items():
        pred_name = tower_name.replace("_tower", "")
        L_task = _tower_lipschitz(tower, training=training, bn_use_moving_stats=bn_use_moving_stats)
        L_heads[pred_name] = float(L_pre_dense * L_user * L_task)

    # ---- utility bound ----
    k_p0 = "paid_treatment_0"
    k_p1 = "paid_treatment_1"
    k_c0 = "cost_treatment_0"
    k_c1 = "cost_treatment_1"
    missing = [k for k in [k_p0, k_p1, k_c0, k_c1] if k not in L_heads]
    if missing:
        raise ValueError(
            "Cannot form utility u1 bound; missing heads: "
            + ", ".join(missing)
            + ". Check your naming."
        )

    L_u1 = (
        L_heads[k_p1] + L_heads[k_p0]
        + float(lambda_cost) * (L_heads[k_c1] + L_heads[k_c0])
    )

    assumptions = {
        "perturb_dense_only": True,
        "dense_feature_count": int(n_dense),
        "dense_std_source": dense_std_source,
        "training_flag_for_dropout": bool(training),
        "bn_uses_moving_stats": bool(bn_use_moving_stats),
        "note_bn_training_mode": "BN global bound in training mode is ill-defined; we use moving stats approximation.",
    }

    return LipschitzBoundReport(
        L_pre_dense=float(L_pre_dense),
        L_user_tower=float(L_user),
        L_heads_logits={k: float(v) for k, v in L_heads.items()},
        L_utility_u1=float(L_u1),
        lambda_cost=float(lambda_cost),
        assumptions=assumptions,
    )


# -----------------------------
# Decision robustness check (margin rule)
# -----------------------------

def decision_robustness_under_epsilon(
    model: tf.keras.Model,
    sample_features: Any,
    epsilon: float,
    lambda_cost: float,
    lipschitz_u1_bound: float,
) -> Dict[str, Any]:
    preds = model(sample_features, training=False)
    U, treatment_ids = Utility.compute_utilities_from_predictions(preds, lambda_cost=lambda_cost)

    U_sorted = np.sort(U, axis=1)
    best = U_sorted[:, -1]
    second = U_sorted[:, -2]
    margin = best - second

    L = float(lipschitz_u1_bound)
    safe_radius = margin / (2.0 * max(L, 1e-12))
    is_robust = safe_radius > float(epsilon)

    best_idx = np.argmax(U, axis=1)
    best_treatment = [treatment_ids[i] for i in best_idx.tolist()]

    return {
        "epsilon": float(epsilon),
        "lambda_cost": float(lambda_cost),
        "lipschitz_upper_bound_u1": float(L),
        "treatment_ids": treatment_ids,
        "best_treatment": best_treatment,
        "decision_margin": margin,
        "safe_radius": safe_radius,
        "is_robust": is_robust,
        "robustness_ratio": float(np.mean(is_robust)),
    }


# -----------------------------
# Main
# -----------------------------

def load_sample_npz(sample_path: str) -> Any:
    data = np.load(sample_path, allow_pickle=True)
    if "x" in data.files:
        return data["x"].astype(np.float32)
    return {k: tf.constant(data[k]) for k in data.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Path to tf.keras saved_model directory")
    ap.add_argument("--sample_npz", type=str, required=True, help="Path to .npz with sample features")
    ap.add_argument("--lambda_cost", type=float, default=0.5, help="lambda for utility u1")
    ap.add_argument("--epsilon", type=float, default=1000.0, help="epsilon for robustness margin check")
    ap.add_argument("--emp_n", type=int, default=1000, help="empirical samples")
    ap.add_argument("--emp_eps", type=float, default=0.01, help="empirical perturbation epsilon")
    ap.add_argument("--std_lower_bound", type=float, default=1e-3, help="fallback std lower bound if no global std")
    ap.add_argument("--dense_keys", type=str, default="", help="comma-separated dense keys for dict input (optional)")
    ap.add_argument("--out_json", type=str, default="robustness_report.json")
    ap.add_argument("--no_empirical", action="store_true", help="skip empirical lipschitz")
    args = ap.parse_args()

    dense_keys = [s.strip() for s in args.dense_keys.split(",") if s.strip()] or None

    print(f"[INFO] Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    print("[INFO] Loading sample features:", args.sample_npz)
    sample_features = load_sample_npz(args.sample_npz)

    # If user did not pass dense_keys and model has dense_feature_names, use them for empirical perturbation
    if dense_keys is None:
        dense_keys = getattr(model, "dense_feature_names", None)

    # 1) Lipschitz upper bound (more complete)
    print("\n========== Lipschitz Upper Bound (Dense-only, tower-aware) ==========")
    bound = compute_model_lipschitz_u1_upper_bound(
        model=model,
        lambda_cost=args.lambda_cost,
        training=False,  # verification typically inference
        std_lower_bound=args.std_lower_bound,
        dense_feature_names=dense_keys if isinstance(dense_keys, list) else None,
        bn_use_moving_stats=True,
    )
    bound_dict = bound.to_dict()
    print(f"[RESULT] L_pre_dense        : {bound.L_pre_dense:.6g}")
    print(f"[RESULT] L_user_tower       : {bound.L_user_tower:.6g}")
    print(f"[RESULT] L_utility_u1 (bound): {bound.L_utility_u1:.6g}")
    # Print heads sorted for readability
    for k in sorted(bound.L_heads_logits.keys()):
        print(f"  - head {k:20s} L <= {bound.L_heads_logits[k]:.6g}")

    report: Dict[str, Any] = {"lipschitz_bound": bound_dict}

    # 2) Empirical Lipschitz
    if not args.no_empirical:
        print("\n========== Empirical Lipschitz (Random perturbation) ==========")
        print(f"[INFO] emp_eps: {args.emp_eps}, emp_n: {args.emp_n}")
        L_emp = empirical_lipschitz_estimate(
            model=model,
            features_batch=sample_features,
            lambda_cost=args.lambda_cost,
            n_samples=args.emp_n,
            epsilon=args.emp_eps,
            dense_keys=dense_keys if isinstance(dense_keys, list) else None,
        )
        print(f"[RESULT] Empirical Lipschitz (utility U): {L_emp:.6g}")
        report["empirical_lipschitz"] = {
            "epsilon": float(args.emp_eps),
            "n_samples": int(args.emp_n),
            "L_empirical": float(L_emp),
            "dense_keys_used": dense_keys if isinstance(dense_keys, list) else None,
        }

    # 3) Decision robustness under epsilon (margin rule with L_u1 bound)
    print("\n========== Decision Robustness under L2 epsilon (margin rule) ==========")
    rob = decision_robustness_under_epsilon(
        model=model,
        sample_features=sample_features,
        epsilon=args.epsilon,
        lambda_cost=args.lambda_cost,
        lipschitz_u1_bound=bound.L_utility_u1,
    )
    print(f"epsilon              : {rob['epsilon']}")
    print(f"L (upper bound u1)   : {rob['lipschitz_upper_bound_u1']:.6g}")
    print(f"robustness_ratio     : {rob['robustness_ratio']:.4f}")
    print(f"best_treatment[:10]  : {rob['best_treatment'][:10]}")
    print(f"margin[:10]          : {rob['decision_margin'][:10]}")
    print(f"safe_radius[:10]     : {rob['safe_radius'][:10]}")
    print(f"is_robust[:10]       : {rob['is_robust'][:10]}")
    report["decision_robustness"] = rob

    # Save report
    def _json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if tf.is_tensor(obj):
            return obj.numpy().tolist()
        return str(obj)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"\n[INFO] Saved report to: {args.out_json}")


if __name__ == "__main__":
    # Make CPU-only optional (uncomment if needed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
