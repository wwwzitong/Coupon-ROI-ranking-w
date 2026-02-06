#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SDP Lipschitz verification (layerwise SDP) + decision robustness check.

Key fix vs your current version:
- For multi-head models, DO NOT multiply all heads serially.
- Instead, compute a Lipschitz bound PER OUTPUT HEAD along its actual graph path,
  then combine them to bound the utility u1 = (paid1-paid0) - lambda*(cost1-cost0).

References:
- Keras Functional API supports DAG models with multiple outputs, and you can reuse
  the same graph to define multiple models/outputs. :contentReference[oaicite:3]{index=3}
- Lipschitz continuity definition/properties. :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cvxpy as cp
import tensorflow as tf


# -----------------------------
# Utilities
# -----------------------------
def _batchnorm_lipschitz(layer: tf.keras.layers.BatchNormalization) -> float:
    """
    Lipschitz constant of BatchNormalization in inference mode (training=False).

    BN inference: y = gamma * (x - moving_mean) / sqrt(moving_var + eps) + beta
    Jacobian wrt x is diagonal with entries scale_i = gamma_i / sqrt(moving_var_i + eps)
    Thus L2 Lipschitz = max_i |scale_i|.
    """
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        raise TypeError("layer must be tf.keras.layers.BatchNormalization")

    if not hasattr(layer, "moving_variance") or layer.moving_variance is None:
        raise ValueError(f"BatchNorm layer {layer.name} seems not built / has no moving_variance.")

    eps = float(getattr(layer, "epsilon", 1e-3))
    mv = layer.moving_variance.numpy().astype(np.float64)

    if getattr(layer, "scale", True) and layer.gamma is not None:
        gamma = layer.gamma.numpy().astype(np.float64)
    else:
        gamma = np.ones_like(mv, dtype=np.float64)

    scale = gamma / np.sqrt(mv + eps)
    return float(np.max(np.abs(scale)))


def _activation_lipschitz(activation: Optional[Union[str, Any]]) -> float:
    """
    Global Lipschitz constant of a pointwise activation in L2.
    For elementwise activations, L2 Lipschitz = max |phi'(z)|.

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
        name = getattr(activation, "__name__", str(activation)).lower()

    if "relu" in name:
        return 1.0
    if "tanh" in name:
        return 1.0
    if "sigmoid" in name:
        return 0.25
    if "linear" in name:
        return 1.0
    return 1.0


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
                return
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"SDP solve failed. Last error: {last_err}, status={prob.status}")


# -----------------------------
# Empirical Lipschitz estimate
# -----------------------------
def _as_numpy_features(x: Any) -> Any:
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
    if isinstance(x_batch_np, dict):
        return {k: v[idx:idx+1] for k, v in x_batch_np.items()}
    return x_batch_np[idx:idx+1]


def _l2_norm_features(delta: Any) -> float:
    if isinstance(delta, dict):
        s = 0.0
        for v in delta.values():
            vv = np.asarray(v)
            s += float(np.sum(vv.astype(np.float64).ravel() ** 2))
        return float(np.sqrt(s))
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
            if ratio > max_ratio:
                max_ratio = ratio

    return float(max_ratio)


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


@dataclass
class HeadLipschitzReport:
    head_name: str
    L_head: float
    layer_details: List[LayerLipschitzResult]
    used_layers: List[str]


class RobustnessSDP:
    """
    Layerwise SDP Lipschitz bound + multi-head safe combination + decision robustness check.
    """

    # ---------- SDP for one Dense ----------
    @staticmethod
    def verify_dense_layer_lipschitz_sdp(
        W: np.ndarray,
        activation: Optional[Union[str, Any]] = "relu",
        p_structure: str = "diag",
        solver: Optional[str] = None,
        verbose: bool = False,
        eps_pd: float = 1e-6,
    ) -> Tuple[float, str]:
        if W.ndim != 2:
            raise ValueError(f"W must be 2D, got shape={W.shape}")

        n_out, n_in = W.shape
        gamma = cp.Variable(nonneg=True)

        if p_structure.lower() == "full":
            P = cp.Variable((n_in, n_in), PSD=True)
            P_mat = P
            P_lb = eps_pd * np.eye(n_in)
        elif p_structure.lower() == "diag":
            p = cp.Variable(n_in, nonneg=True)
            P_mat = cp.diag(p)
            P_lb = eps_pd * np.eye(n_in)
        else:
            raise ValueError("p_structure must be 'diag' or 'full'")

        block = cp.bmat([
            [P_mat,         P_mat @ W.T],
            [W @ P_mat,     gamma * np.eye(n_out)],
        ])

        constraints = [
            block >> 0,
            P_mat - P_lb >> 0,
        ]

        prob = cp.Problem(cp.Minimize(gamma), constraints)
        _try_solve(prob, prefer=solver, verbose=verbose)

        if gamma.value is None:
            raise RuntimeError(f"SDP returned no gamma. status={prob.status}")

        L_linear = float(np.sqrt(max(gamma.value, 0.0) / eps_pd))
        return L_linear, str(prob.status)

    # ---------- Graph tracing (Functional models) ----------
    @staticmethod
    def _is_graph_model(model: tf.keras.Model) -> bool:
        return bool(getattr(model, "_is_graph_network", False))

    @staticmethod
    def _keras_history_unpack(t: tf.Tensor) -> Optional[Tuple[Any, int, int]]:
        """
        Compatible with different Keras versions:
        - t._keras_history may be a tuple (layer, node_index, tensor_index)
        - or an object with attributes .layer/.node_index/.tensor_index (or .operation)
        """
        kh = getattr(t, "_keras_history", None)
        if kh is None:
            return None

        # tuple-like
        if isinstance(kh, (tuple, list)) and len(kh) == 3:
            layer, node_index, tensor_index = kh
            return layer, int(node_index), int(tensor_index)

        # object-like
        layer = getattr(kh, "layer", None)
        if layer is None:
            layer = getattr(kh, "operation", None)
        node_index = getattr(kh, "node_index", 0)
        tensor_index = getattr(kh, "tensor_index", 0)
        if layer is None:
            return None
        return layer, int(node_index), int(tensor_index)

    @staticmethod
    def trace_layers_for_output_tensor(output_tensor: tf.Tensor) -> List[tf.keras.layers.Layer]:
        """
        Trace the layers that contribute to a given output tensor in a Functional graph.
        Returns layers in forward/topological-ish order (inputs -> output).
        """
        layers_postorder: List[tf.keras.layers.Layer] = []
        visited_layers = set()
        visited_tensors = set()

        def rec(t: tf.Tensor):
            if t is None:
                return
            tid = id(t)
            if tid in visited_tensors:
                return
            visited_tensors.add(tid)

            info = RobustnessSDP._keras_history_unpack(t)
            if info is None:
                return
            layer, node_index, _ = info

            # Recurse on inputs to this node
            try:
                node = layer._inbound_nodes[node_index]
                inbound_tensors = node.input_tensors
            except Exception:
                inbound_tensors = getattr(layer, "input", None)

            for it in tf.nest.flatten(inbound_tensors):
                rec(it)

            if id(layer) not in visited_layers:
                visited_layers.add(id(layer))
                layers_postorder.append(layer)

        rec(output_tensor)
        layers_postorder.reverse()
        return layers_postorder

    # ---------- Layerwise Lipschitz along a given layer list ----------
    @staticmethod
    def lipschitz_upper_bound_along_layers(
        layers_in_order: List[tf.keras.layers.Layer],
        p_structure: str = "diag",
        solver: Optional[str] = None,
        verbose: bool = False,
        eps_pd: float = 1e-6,
        layer_filter: Optional[List[str]] = None,
    ) -> Tuple[float, List[LayerLipschitzResult]]:
        allowed = (tf.keras.layers.Dense, tf.keras.layers.BatchNormalization)

        results: List[LayerLipschitzResult] = []
        L_total = 1.0

        for layer in layers_in_order:
            if not isinstance(layer, allowed):
                continue
            if layer_filter is not None and layer.name not in set(layer_filter):
                continue

            if isinstance(layer, tf.keras.layers.Dense):
                W = layer.kernel.numpy()   # (n_in, n_out)
                W_mat = W.T                # (n_out, n_in)

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

            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                L_bn = _batchnorm_lipschitz(layer)
                L_total *= L_bn
                n_chan = int(np.prod(layer.moving_variance.shape))
                results.append(
                    LayerLipschitzResult(
                        layer_name=layer.name,
                        n_in=n_chan,
                        n_out=n_chan,
                        activation="batchnorm",
                        L_linear=L_bn,
                        L_activation=1.0,
                        L_total=L_bn,
                        sdp_status="closed_form",
                    )
                )

        return float(L_total), results

    # ---------- Multi-head: per-output Lipschitz ----------
    @staticmethod
    def per_head_lipschitz_upper_bounds(
        model: tf.keras.Model,
        head_names: List[str],
        p_structure: str = "diag",
        solver: Optional[str] = None,
        verbose: bool = False,
        eps_pd: float = 1e-6,
    ) -> Dict[str, HeadLipschitzReport]:
        """
        Compute Lipschitz bound for each requested head output.

        If model is Functional (graph network), we trace exact contributing layers per head.
        Otherwise fallback to old "flatten layers then multiply" (WARNING).
        """
        reports: Dict[str, HeadLipschitzReport] = {}

        if RobustnessSDP._is_graph_model(model):
            # Map head_name -> output tensor
            # Keras supports dict outputs; in that case model.output_names are the keys.
            name_to_tensor: Dict[str, tf.Tensor] = {}
            try:
                for name, t in zip(model.output_names, model.outputs):
                    name_to_tensor[name] = t
            except Exception:
                # best effort
                pass

            missing = [h for h in head_names if h not in name_to_tensor]
            if missing:
                raise ValueError(
                    "Graph model detected, but cannot find these heads in model.output_names: "
                    + ", ".join(missing)
                    + "\nAvailable output_names: "
                    + ", ".join(getattr(model, "output_names", []))
                )

            for h in head_names:
                out_t = name_to_tensor[h]
                layers_path = RobustnessSDP.trace_layers_for_output_tensor(out_t)
                L_h, details = RobustnessSDP.lipschitz_upper_bound_along_layers(
                    layers_in_order=layers_path,
                    p_structure=p_structure,
                    solver=solver,
                    verbose=verbose,
                    eps_pd=eps_pd,
                )
                reports[h] = HeadLipschitzReport(
                    head_name=h,
                    L_head=float(L_h),
                    layer_details=details,
                    used_layers=[ly.name for ly in layers_path],
                )
            return reports

        # -------- fallback: old behavior (not accurate for multi-head) --------
        print(
            "[WARN] Model is NOT a graph network (likely pure subclassed model). "
            "Cannot trace per-head paths reliably; falling back to multiplying all Dense/BN layers. "
            "For accurate multi-head bounds, consider exporting/building the model using Functional API "
            "or Functional-Subclassing pattern."
        )

        # Flatten all layers
        layers_all = list(model._flatten_layers(include_self=False, recursive=True))
        for h in head_names:
            L_h, details = RobustnessSDP.lipschitz_upper_bound_along_layers(
                layers_in_order=layers_all,
                p_structure=p_structure,
                solver=solver,
                verbose=verbose,
                eps_pd=eps_pd,
            )
            reports[h] = HeadLipschitzReport(
                head_name=h,
                L_head=float(L_h),
                layer_details=details,
                used_layers=[ly.name for ly in layers_all],
            )
        return reports

    # ---------- Utility Lipschitz (your incremental decision) ----------
    @staticmethod
    def utility_lipschitz_upper_bound_incremental(
        per_head: Dict[str, HeadLipschitzReport],
        paid_prefix: str = "paid_treatment_",
        cost_prefix: str = "cost_treatment_",
        lambda_cost: float = 0.5,
    ) -> float:
        """
        u1 = (paid1 - paid0) - lambda*(cost1 - cost0)

        Using triangle inequality:
        Lip(u1) <= Lip(paid1)+Lip(paid0)+lambda*(Lip(cost1)+Lip(cost0))
        """
        k_p0 = f"{paid_prefix}0"
        k_p1 = f"{paid_prefix}1"
        k_c0 = f"{cost_prefix}0"
        k_c1 = f"{cost_prefix}1"

        for k in [k_p0, k_p1, k_c0, k_c1]:
            if k not in per_head:
                raise ValueError(f"Missing head in per_head report: {k}")

        L = (
            per_head[k_p1].L_head
            + per_head[k_p0].L_head
            + float(lambda_cost) * (per_head[k_c1].L_head + per_head[k_c0].L_head)
        )
        return float(L)

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
    def compute_utilities_from_predictions(
        preds: Any,
        paid_prefix: str = "paid_treatment_",
        cost_prefix: str = "cost_treatment_",
        lambda_cost: float = 0.5,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Binary decision utility:
        u1 = (paid_1 - paid_0) - lambda * (cost_1 - cost_0)
        return U=[0,u1] so argmax matches threshold rule.
        """
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

    @staticmethod
    def decision_robustness_under_epsilon(
        model: tf.keras.Model,
        sample_features: Any,
        epsilon: float = 0.1,
        lipschitz_bound: Optional[float] = None,
        per_head_reports: Optional[Dict[str, HeadLipschitzReport]] = None,
        lambda_cost: float = 0.5,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Conservative guarantee:
          If margin(x) > 2 * L * epsilon, argmax cannot change.
        Here U=[0,u1], margin = |u1|.
        """
        preds = model(sample_features, training=False)
        U, treatment_ids = RobustnessSDP.compute_utilities_from_predictions(
            preds, lambda_cost=lambda_cost
        )

        # Since U=[0,u1], margin = |u1|
        u1 = U[:, 1]
        margin = np.abs(u1)

        if lipschitz_bound is None:
            if per_head_reports is None:
                raise ValueError("Provide either lipschitz_bound or per_head_reports.")
            L_net = RobustnessSDP.utility_lipschitz_upper_bound_incremental(
                per_head=per_head_reports,
                lambda_cost=lambda_cost,
            )
        else:
            L_net = float(lipschitz_bound)

        safe_radius = margin / (2.0 * max(L_net, 1e-12))
        is_robust = safe_radius > float(epsilon)

        best_idx = np.argmax(U, axis=1)
        best_treatment = [treatment_ids[i] for i in best_idx.tolist()]

        out = {
            "epsilon": float(epsilon),
            "lipschitz_upper_bound": float(L_net),
            "treatment_ids": treatment_ids,
            "best_treatment": best_treatment,
            "decision_margin": margin,
            "safe_radius": safe_radius,
            "is_robust": is_robust,
            "robustness_ratio": float(np.mean(is_robust)),
        }
        if per_head_reports is not None:
            out["per_head"] = {
                k: {
                    "L_head": float(v.L_head),
                    "used_layers": v.used_layers,
                    "layer_details": [r.__dict__ for r in v.layer_details],
                }
                for k, v in per_head_reports.items()
            }
        if verbose:
            print("[DEBUG] L used for robustness:", L_net)
        return out


# -----------------------------
# Example main (edit to your environment)
# -----------------------------
def main():
    print("[INFO] Loading model")
    model = tf.keras.models.load_model(
        "../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40",
        compile=False
    )
    model.summary()

    # ===== Load sample features =====
    sample_path = "sample_features_ECLIFT.npz"
    print(f"\n[INFO] Loading sample features from: {sample_path}")
    data = np.load(sample_path, allow_pickle=True)
    if "x" in data.files:
        sample_features = data["x"].astype(np.float32)
    else:
        sample_features = {k: tf.constant(data[k]) for k in data.files}

    # ===== Multi-head per-output Lipschitz via SDP =====
    head_names = [
        "paid_treatment_0",
        "paid_treatment_1",
        "cost_treatment_0",
        "cost_treatment_1",
    ]

    print("\n[INFO] Computing per-head Lipschitz upper bounds (SDP along each head path)")
    per_head = RobustnessSDP.per_head_lipschitz_upper_bounds(
        model=model,
        head_names=head_names,
        p_structure="diag",     # "diag" faster; "full" tighter but heavy
        solver="SCS",         # change if not available
        verbose=False,
        eps_pd=1e-6,
    )

    for h, rep in per_head.items():
        print(f"\n========== Head: {h} ==========")
        print(f"[RESULT] L_head <= {rep.L_head:.6g}")
        # Print a compact layer table (only Dense/BN contribute)
        for r in rep.layer_details:
            print(
                f"- {r.layer_name:30s} "
                f"({r.n_in:4d}->{r.n_out:4d}, act={r.activation:7s}) "
                f"L_lin={r.L_linear:.4g}  L_act={r.L_activation:.4g}  "
                f"L_layer={r.L_total:.4g}  status={r.sdp_status}"
            )

    # ===== Utility Lipschitz (what you really need for decision robustness) =====
    lambda_cost = 0.5
    L_u1 = RobustnessSDP.utility_lipschitz_upper_bound_incremental(
        per_head=per_head,
        lambda_cost=lambda_cost,
    )
    print("\n========== Utility Lipschitz Upper Bound ==========")
    print(f"[RESULT] For u1=(paid1-paid0)-{lambda_cost}*(cost1-cost0):  L_u1 <= {L_u1:.6g}")

    # ===== Empirical Lipschitz estimate =====
    emp_L = empirical_lipschitz_estimate(
        model=model,
        x_batch=sample_features,
        n_samples=1000,
        epsilon=0.01,
        seed=42,
    )
    print("\n========== Empirical Lipschitz Estimate ==========")
    print(f"[RESULT] Empirical Lipschitz (epsilon=0.01, n_samples=1000): ~ {emp_L:.6g}")
    if emp_L > 0:
        print(f"[INFO] (L_u1 upper bound) / empirical ~= {L_u1 / emp_L:.6g}")

    # ===== Decision robustness =====
    report = RobustnessSDP.decision_robustness_under_epsilon(
        model=model,
        sample_features=sample_features,
        epsilon=0.1,
        lipschitz_bound=L_u1,        # use the corrected utility Lipschitz bound
        per_head_reports=per_head,   # also store details
        lambda_cost=lambda_cost,
        verbose=False,
    )

    print("\n========== Decision Robustness under L2 epsilon ==========")
    print(f"epsilon              : {report['epsilon']}")
    print(f"L (utility upper)    : {report['lipschitz_upper_bound']:.6g}")
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
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if tf.is_tensor(obj):
            return obj.numpy().tolist()
        return str(obj)

    out_json = os.environ.get("OUT_JSON", "sdp_lipschitz_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\n[INFO] Saved report to: {out_json}")


if __name__ == "__main__":
    main()
