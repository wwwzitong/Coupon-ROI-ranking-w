#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SDP Lipschitz verification (layerwise SDP) + decision robustness check.

Dependencies:
  pip install cvxpy numpy tensorflow

Notes:
- This script computes a Lipschitz upper bound for each Dense layer via SDP,
  then multiplies them to get a network upper bound.
- Bias does NOT affect Lipschitz constant and is ignored.
- Activations supported: relu / sigmoid / tanh / linear (or None)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cvxpy as cp
import tensorflow as tf


# -----------------------------
# Utilities
# -----------------------------

def _activation_lipschitz(activation: Optional[Union[str, Any]]) -> float:
    """
    Return a global Lipschitz constant of a pointwise activation in L2.
    For elementwise activations, the L2 Lipschitz is max |phi'(z)|.

    - ReLU: 1
    - Tanh: 1
    - Sigmoid: 0.25 (max derivative at 0)
    - Linear/None: 1
    """
    if activation is None:
        return 1.0
    if isinstance(activation, str):
        name = activation.lower()
    else:
        # Keras activation function or layer activation
        name = getattr(activation, "__name__", str(activation)).lower()

    if "relu" in name:
        return 1.0
    if "tanh" in name:
        return 1.0
    if "sigmoid" in name:
        return 0.25
    if "linear" in name:
        return 1.0

    # Unknown activation: fall back conservatively to 1.0
    return 1.0


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
                print(f"solver: {s}")
                return
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"SDP solve failed. Last error: {last_err}, status={prob.status}")


# -----------------------------
# SDP core
# -----------------------------

@dataclass
class LayerLipschitzResult:
    layer_name: str
    n_in: int
    n_out: int
    activation: str
    L_linear: float
    L_activation: float
    L_total: float
    sdp_status: str


class RobustnessSDP:
    """
    Layerwise SDP Lipschitz bound + network bound + decision robustness check.
    """

    @staticmethod
    def verify_dense_layer_lipschitz_sdp(
        W: np.ndarray,
        activation: Optional[Union[str, Any]] = "relu",
        p_structure: str = "diag",
        solver: Optional[str] = None,
        verbose: bool = False,
        eps_pd: float = 1e-6,
    ) -> Tuple[float, str]:
        """
        Compute a Lipschitz upper bound of a *linear map* y = W x using SDP,
        then multiply by activation Lipschitz outside this function.

        SDP (Schur-complement style):
            minimize gamma
            s.t.  [ P,      P W^T ]
                  [ W P,  gamma I ]  >= 0
                  P >= eps I

        This implies ||W||_2 <= gamma / eps (in a conservative way depending on scaling),
        but in practice gives a valid upper bound and is often tighter than naive bounds
        when using diagonal P for large layers.

        Args:
            W: (n_out, n_in) weight matrix
            activation: activation after this Dense layer (used outside for scaling)
            p_structure: "diag" (scalable) or "full" (tighter but heavy)
            solver: preferred cvxpy solver name (e.g. "SCS", "MOSEK")
        Returns:
            (L_linear_upper_bound, sdp_status)
        """
        if W.ndim != 2:
            raise ValueError(f"W must be 2D, got shape={W.shape}")

        n_out, n_in = W.shape

        gamma = cp.Variable(nonneg=True)

        if p_structure.lower() == "full":
            P = cp.Variable((n_in, n_in), PSD=True)
            P_mat = P
            P_lb = eps_pd * np.eye(n_in)
        elif p_structure.lower() == "diag":
            p = cp.Variable(n_in, nonneg=True)  # diagonal entries
            P_mat = cp.diag(p)
            P_lb = eps_pd * np.eye(n_in)
        else:
            raise ValueError("p_structure must be 'diag' or 'full'")

        # LMI block
        # NOTE: Use numpy arrays for constants; W is constant.
        block = cp.bmat([
            [P_mat,            P_mat @ W.T],
            [W @ P_mat,        gamma * np.eye(n_out)],
        ])

        constraints = [
            block >> 0,
            P_mat - P_lb >> 0,
        ]

        prob = cp.Problem(cp.Minimize(gamma), constraints)
        _try_solve(prob, prefer=solver, verbose=verbose)

        if gamma.value is None:
            raise RuntimeError(f"SDP returned no gamma. status={prob.status}")

        # A usable bound for ||W||_2:
        # Under P >= eps I and [P, P W^T; W P, gamma I] >= 0, one can derive
        # W P W^T <= gamma I  =>  ||W||_2^2 <= gamma / eps.
        # (Conservative but valid)
        L_linear = float(np.sqrt(max(gamma.value, 0.0) / eps_pd))
        return L_linear, str(prob.status)

    @staticmethod
    def extract_dense_layers(model: tf.keras.Model) -> List[tf.keras.layers.Dense]:
        """
        Extract all tf.keras.layers.Dense layers from a (possibly nested) Keras model.
        """
        dense_layers: List[tf.keras.layers.Dense] = []

        def _walk(layer: tf.keras.layers.Layer):
            if isinstance(layer, tf.keras.layers.Dense):
                dense_layers.append(layer)
            # nested model/layer containers
            if hasattr(layer, "layers") and isinstance(layer.layers, list):
                for sub in layer.layers:
                    _walk(sub)

        _walk(model)
        return dense_layers

    @staticmethod
    def network_lipschitz_upper_bound(
        model: tf.keras.Model,
        p_structure: str = "diag",
        solver: Optional[str] = None,
        verbose: bool = False,
        include_bias: bool = False,  # kept for API clarity; ignored
        layer_filter: Optional[List[str]] = None,
        eps_pd: float = 1e-6,
    ) -> Tuple[float, List[LayerLipschitzResult]]:
        """
        Compute a network Lipschitz upper bound by multiplying layerwise SDP bounds.

        Args:
            layer_filter: if provided, only include Dense layers whose .name is in this list.
        """
        dense_layers = RobustnessSDP.extract_dense_layers(model)
        if layer_filter is not None:
            dense_layers = [l for l in dense_layers if l.name in set(layer_filter)]

        results: List[LayerLipschitzResult] = []
        L_total = 1.0

        for layer in dense_layers:
            W = layer.kernel.numpy()  # (n_in, n_out) in Keras
            # Convert to (n_out, n_in)
            W_mat = W.T

            act = layer.activation
            act_name = getattr(act, "__name__", "linear")
            L_act = _activation_lipschitz(act)

            L_lin, status = RobustnessSDP.verify_dense_layer_lipschitz_sdp(
                W=W_mat,
                activation=act_name,
                p_structure=p_structure,
                solver=solver,
                verbose=verbose,
                eps_pd=eps_pd,
            )

            L_layer_total = L_lin * L_act
            L_total *= L_layer_total

            results.append(
                LayerLipschitzResult(
                    layer_name=layer.name,
                    n_in=W_mat.shape[1],
                    n_out=W_mat.shape[0],
                    activation=str(act_name),
                    L_linear=L_lin,
                    L_activation=L_act,
                    L_total=L_layer_total,
                    sdp_status=status,
                )
            )

        return float(L_total), results

    # -----------------------------
    # Decision robustness
    # -----------------------------

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if tf.is_tensor(x):
            return x.numpy()
        return np.asarray(x)

    @staticmethod
    def compute_utilities_from_predictions_old(
        preds: Any,
        paid_prefix: str = "paid_treatment_",
        cost_prefix: str = "cost_treatment_",
        utility_mode: str = "paid_minus_cost",
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Convert model predictions to a utility matrix U of shape (batch, K),
        where K is number of treatments/actions.

        Expected preds: dict-like with keys such as:
          paid_treatment_0, cost_treatment_0, paid_treatment_1, cost_treatment_1, ...

        Returns:
          U: (B, K)
          treatments: sorted list of treatment indices found
        """
        if not isinstance(preds, dict):
            raise ValueError("Model output must be a dict for compute_utilities_from_predictions().")

        paid_keys = [k for k in preds.keys() if k.startswith(paid_prefix)]
        cost_keys = [k for k in preds.keys() if k.startswith(cost_prefix)]

        def _idx(k: str, prefix: str) -> int:
            return int(k.replace(prefix, ""))

        paid_idx = { _idx(k, paid_prefix): k for k in paid_keys }
        cost_idx = { _idx(k, cost_prefix): k for k in cost_keys }

        common = sorted(set(paid_idx.keys()) & set(cost_idx.keys()))
        if len(common) == 0:
            raise ValueError(
                f"No common treatment heads found. Need both {paid_prefix}* and {cost_prefix}*."
            )

        utils = []
        for t in common:
            paid = RobustnessSDP._to_numpy(preds[paid_idx[t]]).reshape(-1)
            cost = RobustnessSDP._to_numpy(preds[cost_idx[t]]).reshape(-1)
            if utility_mode == "paid_minus_cost":
                u = paid - cost
            elif utility_mode == "paid_over_cost":
                u = paid / (np.maximum(cost, 1e-12))
            else:
                raise ValueError("utility_mode must be 'paid_minus_cost' or 'paid_over_cost'")
            utils.append(u)

        U = np.stack(utils, axis=1)  # (B, K)
        return U, common

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

    @staticmethod
    def decision_robustness_under_epsilon(
        model: tf.keras.Model,
        sample_features: Any,
        epsilon: float = 0.1,
        lipschitz_bound: Optional[float] = None,
        p_structure: str = "diag",
        solver: Optional[str] = None,
        verbose: bool = False,
        eps_pd: float = 1e-6,
        utility_mode: str = "paid_minus_cost",
    ) -> Dict[str, Any]:
        """
        Check if argmax decision over utilities is robust under ||delta x||_2 <= epsilon.

        Conservative guarantee:
          If margin(x) > 2 * L * epsilon,
          where margin(x) = u_best(x) - u_second(x),
          then the argmax cannot change under L-Lipschitz perturbations.

        Args:
            lipschitz_bound: if None, compute via network_lipschitz_upper_bound().
        """
        preds = model(sample_features, training=False)

        U, treatment_ids = RobustnessSDP.compute_utilities_from_predictions(
            preds
        )

        # margin between top-1 and top-2 utility
        U_sorted = np.sort(U, axis=1)
        best = U_sorted[:, -1]
        second = U_sorted[:, -2] if U.shape[1] >= 2 else np.full_like(best, -np.inf)
        margin = best - second  # (B,)

        if lipschitz_bound is None:
            L_net, layer_details = RobustnessSDP.network_lipschitz_upper_bound(
                model=model,
                p_structure=p_structure,
                solver=solver,
                verbose=verbose,
                eps_pd=eps_pd,
            )
        else:
            L_net = float(lipschitz_bound)
            layer_details = []

        safe_radius = margin / (2.0 * max(L_net, 1e-12))
        is_robust = safe_radius > epsilon

        # decisions
        best_idx = np.argmax(U, axis=1)
        best_treatment = [treatment_ids[i] for i in best_idx.tolist()]

        return {
            "epsilon": float(epsilon),
            "lipschitz_upper_bound": float(L_net),
            "utility_mode": utility_mode,
            "treatment_ids": treatment_ids,
            "best_treatment": best_treatment,
            "decision_margin": margin,                 # (B,)
            "safe_radius": safe_radius,               # (B,)
            "is_robust": is_robust,                   # (B,)
            "robustness_ratio": float(np.mean(is_robust)),
            "layer_details": [r.__dict__ for r in layer_details],
        }


# -----------------------------
# Example main (edit to your environment)
# -----------------------------

def main():
    # ===== 1) Load your trained model =====
    print(f"[INFO] Loading model")
    model = tf.keras.models.load_model("../final-criteo/model/EcomDFCL_regretNet_rplusc_wce_batchmean_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5", compile=False)
    # model = tf.keras.models.load_model("../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0", compile=False)
    # model = tf.keras.models.load_model("../final-ECLIFT/model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5", compile=False)
    model.summary()

    # ===== 2) Compute Lipschitz upper bound (layerwise SDP) =====
    # For large layers, "diag" is much faster & more stable.
    L_net, layer_results = RobustnessSDP.network_lipschitz_upper_bound(
        model=model,
        p_structure="diag",    # "full" if small dims and want tighter but slower
        solver="SCS",          # change to "MOSEK" if available
        verbose=False,
        eps_pd=1e-6,
    )

    print("\n========== Lipschitz Upper Bound (Layerwise SDP) ==========")
    print(f"[RESULT] Network Lipschitz upper bound: L <= {L_net:.6g}")
    for r in layer_results:
        print(
            f"- {r.layer_name:30s} "
            f"({r.n_in:4d}->{r.n_out:4d}, act={r.activation:7s}) "
            f"L_lin={r.L_linear:.4g}  L_act={r.L_activation:.4g}  L_layer={r.L_total:.4g}  status={r.sdp_status}"
        )

    # ===== 3) Decision robustness check on a batch of samples =====
    # sample_features must match your model's input signature:
    # - If your model input is a tensor: sample_features = np.random.randn(B, d).astype(np.float32)
    # - If your model input is a dict (common in ranking / wide&deep): provide dict of tensors.
    #
    # Here we provide a placeholder that you MUST replace with real data batch.
    # Example:
    # sample_features = {
    #   "dense": tf.constant(dense_batch, tf.float32),
    #   "sparse": tf.constant(sparse_batch, tf.int32),
    # }
        
    # sample_path = "sample_features_ECLIFT.npz"
    sample_path = "sample_features_criteo.npz"

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

    report = RobustnessSDP.decision_robustness_under_epsilon(
        model=model,
        sample_features=sample_features,
        epsilon=0.1,
        lipschitz_bound=L_net,     # reuse computed bound; or set None to recompute
        p_structure="diag",
        solver="SCS",
        verbose=False,
        utility_mode="paid_minus_cost",
    )

    print("\n========== Decision Robustness under L2 epsilon ==========")
    print(f"epsilon              : {report['epsilon']}")
    print(f"L (upper bound)      : {report['lipschitz_upper_bound']:.6g}")
    print(f"robustness_ratio     : {report['robustness_ratio']:.4f}")
    print(f"treatment_ids        : {report['treatment_ids']}")
    print(f"best_treatment[:10]  : {report['best_treatment'][:10]}")
    print(f"margin[:10]          : {report['decision_margin'][:10]}")
    print(f"safe_radius[:10]     : {report['safe_radius'][:10]}")
    print(f"is_robust[:10]       : {report['is_robust'][:10]}")

    def _json_default(obj):
        import numpy as np
        import tensorflow as tf

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_ ,)):
            return bool(obj)
        if tf.is_tensor(obj):
            return obj.numpy().tolist()
        return str(obj)  # 兜底：至少不崩

    # Save full report
    out_json = os.environ.get("OUT_JSON", "sdp_lipschitz_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        # json.dump(report, f, ensure_ascii=False, indent=2)
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\n[INFO] Saved report to: {out_json}")


if __name__ == "__main__":
    main()
