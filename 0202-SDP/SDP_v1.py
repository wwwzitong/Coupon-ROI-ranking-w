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

def _slice_one_sample(features: Any, idx: int) -> Any:
    """从 batch 里取第 idx 个样本，保持输入结构（tensor 或 dict）。"""
    if isinstance(features, dict):
        out = {}
        for k, v in features.items():
            v_np = v.numpy() if tf.is_tensor(v) else np.asarray(v)
            out[k] = v_np[idx:idx+1]
        return out
    else:
        x_np = features if isinstance(features, np.ndarray) else np.asarray(features)
        return x_np[idx:idx+1]


def _add_delta(features: Any, delta: Any) -> Any:
    """features + delta，保持结构一致；对非数值字段（bytes/str/object）不做加法。"""
    if isinstance(features, dict):
        out = {}
        for k in features.keys():
            v = features[k]
            d = delta.get(k, None) if isinstance(delta, dict) else None

            # 没有 delta 或者 delta 为 None：直接原样复制
            if d is None:
                out[k] = v
                continue

            v_np = v.numpy() if tf.is_tensor(v) else np.asarray(v)

            # 非数值类型：直接原样复制（避免 bytes + float）
            if v_np.dtype.kind not in ("f", "i", "u"):  # float/int/uint 才允许加
                out[k] = v
                continue

            # 数值类型：执行加法
            out[k] = v_np + np.asarray(d, dtype=v_np.dtype)
        return out
    else:
        return features + delta



def _random_delta_like(x: Any, epsilon: float) -> Any:
    """
    生成与 x 同结构的随机扰动 delta，并使整体 L2 范数为 epsilon。
    - tensor 输入：对所有维度扰动
    - dict 输入：只对 float dtype 的键扰动（int/ids 不扰动）
    """
    if isinstance(x, dict):
        deltas = {}
        flats = []

        for k, v in x.items():
            v_np = v.numpy() if tf.is_tensor(v) else np.asarray(v)
            # 只扰动连续特征（float）
            if v_np.dtype.kind in ("f",):  # float32/float64
                d = np.random.randn(*v_np.shape).astype(v_np.dtype)
            else:
                d = np.zeros_like(v_np)
            deltas[k] = d
            flats.append(d.reshape(-1))

        flat = np.concatenate(flats, axis=0)
        norm = np.linalg.norm(flat)
        if norm < 1e-12:
            # 没有可扰动的 float 特征（或全 0），直接返回全 0
            return deltas

        scale = epsilon / norm
        for k in deltas.keys():
            deltas[k] = deltas[k] * scale
        return deltas
    else:
        x_np = x if isinstance(x, np.ndarray) else np.asarray(x)
        d = np.random.randn(*x_np.shape).astype(x_np.dtype)
        flat = d.reshape(-1)
        norm = np.linalg.norm(flat)
        if norm < 1e-12:
            return np.zeros_like(x_np)
        return d / norm * epsilon


def empirical_lipschitz_estimate(
    model: tf.keras.Model,
    features_batch: Any,
    n_samples: int = 1000,
    epsilon: float = 0.01,
) -> float:
    """
    通过随机扰动估计经验 Lipschitz 常数（针对 utility U 的变化）。
    返回采样到的 max ||U(x)-U(x+δ)|| / ||δ||。
    支持输入为 tensor 或 dict。
    """
    # batch size
    if isinstance(features_batch, dict):
        # 任取一个 key 获取 batch size
        any_key = next(iter(features_batch.keys()))
        B = (features_batch[any_key].shape[0] if tf.is_tensor(features_batch[any_key])
             else np.asarray(features_batch[any_key]).shape[0])
    else:
        B = len(features_batch)

    max_ratio = 0.0

    for _ in range(n_samples):
        idx = np.random.randint(0, B)
        x = _slice_one_sample(features_batch, idx)
        delta = _random_delta_like(x, epsilon=epsilon)

        x1 = x
        x2 = _add_delta(x, delta)

        pred1 = model(x1, training=False)
        pred2 = model(x2, training=False)

        u1, _ = RobustnessSDP.compute_utilities_from_predictions(pred1)
        u2, _ = RobustnessSDP.compute_utilities_from_predictions(pred2)

        output_diff = np.linalg.norm(u1 - u2)
        # input_diff 应该接近 epsilon；但 dict 情况我们也严格算一遍
        if isinstance(delta, dict):
            flat = np.concatenate([np.asarray(delta[k]).reshape(-1) for k in delta.keys()], axis=0)
            input_diff = np.linalg.norm(flat)
        else:
            input_diff = np.linalg.norm(np.asarray(delta))

        if input_diff < 1e-12:
            continue

        ratio = float(output_diff / input_diff)
        if ratio > max_ratio:
            max_ratio = ratio

    return float(max_ratio)

def _is_linear_activation(act) -> bool:
    """Dense activation 是否为 linear/None。"""
    if act is None:
        return True
    name = getattr(act, "__name__", str(act)).lower()
    return ("linear" in name)

def _bn_inference_scale(bn: tf.keras.layers.BatchNormalization) -> np.ndarray:
    """
    推理模式下 BN 的逐维缩放系数 alpha = gamma / sqrt(moving_var + eps)
    返回 shape: (d,)
    """
    # BN 可能没有 scale（gamma），例如 scale=False
    if bn.gamma is None:
        gamma = np.ones_like(bn.moving_variance.numpy(), dtype=np.float32)
    else:
        gamma = bn.gamma.numpy()

    mv = bn.moving_variance.numpy()
    eps = float(bn.epsilon)
    alpha = gamma / np.sqrt(mv + eps)
    return alpha.astype(np.float32)

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
                # print(f"solver: {s}")
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
    # def network_lipschitz_upper_bound(
    #     model: tf.keras.Model,
    #     p_structure: str = "diag",
    #     solver: Optional[str] = None,
    #     verbose: bool = False,
    #     include_bias: bool = False,  # kept for API clarity; ignored
    #     layer_filter: Optional[List[str]] = None,
    #     eps_pd: float = 1e-6,
    # ) -> Tuple[float, List[LayerLipschitzResult]]:
    #     """
    #     Compute a network Lipschitz upper bound by multiplying layerwise SDP bounds.

    #     Args:
    #         layer_filter: if provided, only include Dense layers whose .name is in this list.
    #     """
    #     dense_layers = RobustnessSDP.extract_dense_layers(model)
    #     if layer_filter is not None:
    #         dense_layers = [l for l in dense_layers if l.name in set(layer_filter)]

    #     results: List[LayerLipschitzResult] = []
    #     L_total = 1.0

    #     for layer in dense_layers:
    #         W = layer.kernel.numpy()  # (n_in, n_out) in Keras
    #         # Convert to (n_out, n_in)
    #         W_mat = W.T

    #         act = layer.activation
    #         act_name = getattr(act, "__name__", "linear")
    #         L_act = _activation_lipschitz(act)

    #         L_lin, status = RobustnessSDP.verify_dense_layer_lipschitz_sdp(
    #             W=W_mat,
    #             activation=act_name,
    #             p_structure=p_structure,
    #             solver=solver,
    #             verbose=verbose,
    #             eps_pd=eps_pd,
    #         )

    #         L_layer_total = L_lin * L_act
    #         L_total *= L_layer_total

    #         results.append(
    #             LayerLipschitzResult(
    #                 layer_name=layer.name,
    #                 n_in=W_mat.shape[1],
    #                 n_out=W_mat.shape[0],
    #                 activation=str(act_name),
    #                 L_linear=L_lin,
    #                 L_activation=L_act,
    #                 L_total=L_layer_total,
    #                 sdp_status=status,
    #             )
    #         )

    #     return float(L_total), results

    # -----------------------------
    # Decision robustness
    # -----------------------------

    # @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if tf.is_tensor(x):
            return x.numpy()
        return np.asarray(x)

    # @staticmethod
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

    # @staticmethod
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

    @staticmethod
    def _flatten_layers_in_order(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
        """
        递归展开子层，尽量保持“前向书写顺序”。
        对 Sequential/大多数子模块堆叠的模型很好用。
        Functional 图严格拓扑顺序需要更复杂的 graph walk，这里先不做。
        """
        flat: List[tf.keras.layers.Layer] = []

        def _walk(layer: tf.keras.layers.Layer):
            if hasattr(layer, "layers") and isinstance(layer.layers, list) and layer.layers:
                for sub in layer.layers:
                    _walk(sub)
            else:
                flat.append(layer)

        _walk(model)
        return flat

    @staticmethod
    def network_lipschitz_upper_bound(
        model: tf.keras.Model,
        p_structure: str = "diag",
        solver: Optional[str] = None,
        verbose: bool = False,
        include_bias: bool = False,  # ignored
        layer_filter: Optional[List[str]] = None,
        eps_pd: float = 1e-6,
        merge_dense_bn: bool = True,   # 新增开关：是否启用方案B
    ) -> Tuple[float, List[LayerLipschitzResult]]:
        """
        计算网络 Lipschitz 上界：逐层相乘。
        方案B：若检测到 Dense(linear) -> BatchNorm，则合并成等效线性层后做 SDP。
        """
        layers = RobustnessSDP._flatten_layers_in_order(model)

        # 如果提供 layer_filter：仅保留 name 在其中的层（注意：合并时可能跳过 BN）
        if layer_filter is not None:
            allow = set(layer_filter)
            layers = [l for l in layers if l.name in allow]

        results: List[LayerLipschitzResult] = []
        L_total = 1.0

        i = 0
        while i < len(layers):
            layer = layers[i]

            # --------- 1) Dense（可能与 BN 合并）---------
            if isinstance(layer, tf.keras.layers.Dense):
                # Keras Dense.kernel: (n_in, n_out)
                W = layer.kernel.numpy()
                W_mat = W.T  # (n_out, n_in)

                act = layer.activation
                act_name = getattr(act, "__name__", "linear")
                # 只有 Dense 是线性激活时，才允许合并（否则 BN 在非线性后面，不能并成一个线性算子）
                can_merge = (
                    merge_dense_bn
                    and _is_linear_activation(act)
                    and (i + 1 < len(layers))
                    and isinstance(layers[i + 1], tf.keras.layers.BatchNormalization)
                )

                if can_merge:
                    bn = layers[i + 1]
                    alpha = _bn_inference_scale(bn)  # (n_out,) 理论上应与 Dense 输出维一致

                    if alpha.shape[0] != W_mat.shape[0]:
                        raise ValueError(
                            f"Shape mismatch for merging Dense+BN: "
                            f"Dense out={W_mat.shape[0]}, BN scale dim={alpha.shape[0]} "
                            f"(Dense={layer.name}, BN={bn.name})"
                        )

                    # 等效权重：W_eff = diag(alpha) @ W_mat
                    # 等价实现：按行缩放 W_mat
                    W_eff = (alpha.reshape(-1, 1) * W_mat)

                    # SDP 上界（线性部分）
                    L_lin, status = RobustnessSDP.verify_dense_layer_lipschitz_sdp(
                        W=W_eff,
                        activation="linear",
                        p_structure=p_structure,
                        solver=solver,
                        verbose=verbose,
                        eps_pd=eps_pd,
                    )

                    # 合并块之后的激活不在 Dense 上，需要看 BN 后面是否有单独的 Activation 层
                    # 这里不猜测图结构：只对“紧随其后的显式 Activation/ReLU”等层单独乘
                    # Dense+BN 合并块本身不乘 activation（因为 linear）
                    L_act = 1.0
                    L_layer_total = L_lin * L_act
                    L_total *= L_layer_total

                    results.append(
                        LayerLipschitzResult(
                            layer_name=f"{layer.name}+{bn.name}",
                            n_in=W_eff.shape[1],
                            n_out=W_eff.shape[0],
                            activation="linear",
                            L_linear=L_lin,
                            L_activation=L_act,
                            L_total=L_layer_total,
                            sdp_status=status,
                        )
                    )

                    # 跳过 BN
                    i += 2
                    continue

                else:
                    # 不合并：按你原逻辑 Dense SDP + Dense 内置 activation lipschitz
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

                    i += 1
                    continue

            # --------- 2) 显式 Activation / ReLU / LeakyReLU 等（可选：补齐你之前漏算的）---------
            # 如果你模型里激活是单独一层（而不是写在 Dense.activation 里），建议把它也乘进去。
            if isinstance(layer, (tf.keras.layers.Activation, tf.keras.layers.ReLU,
                                  tf.keras.layers.LeakyReLU, tf.keras.layers.ELU,
                                  tf.keras.layers.PReLU)):
                # 统一用你已有的 _activation_lipschitz
                act = getattr(layer, "activation", None)
                # ReLU 类层 activation 可能没有 __name__，用类名兜底
                if act is None:
                    act_name = layer.__class__.__name__.lower()
                    L_act = _activation_lipschitz(act_name)
                else:
                    act_name = getattr(act, "__name__", layer.__class__.__name__.lower())
                    L_act = _activation_lipschitz(act)

                # 纯激活层没有线性权重
                L_total *= L_act
                results.append(
                    LayerLipschitzResult(
                        layer_name=layer.name,
                        n_in=-1,
                        n_out=-1,
                        activation=str(act_name),
                        L_linear=1.0,
                        L_activation=L_act,
                        L_total=L_act,
                        sdp_status="N/A",
                    )
                )
                i += 1
                continue

            # --------- 3) BN（如果没被合并，可选择把它当逐维缩放乘进去；但这属于方案A，不是B）---------
            # 方案B强调“合并”，这里默认跳过（你也可以改成乘进去）。
            i += 1

        return float(L_total), results
# -----------------------------
# Example main (edit to your environment)
# -----------------------------

def main():
    # ===== 1) Load your trained model =====
    print(f"[INFO] Loading model")
    # model = tf.keras.models.load_model("../final-criteo/model/EcomDFCL_regretNet_rplusc_wce_batchmean_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5", compile=False)
    model = tf.keras.models.load_model("../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0", compile=False)
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

    # ===== 3) Empirical Lipschitz estimate (random perturbation) =====
    # 经验 Lipschitz 是“局部/数据分布附近”的最大斜率采样值，不是严格上界。
    emp_eps = float(os.environ.get("EMP_EPS", "0.01"))
    emp_n = int(os.environ.get("EMP_N", "1000"))

    print("\n========== Empirical Lipschitz (Random Perturbation) ==========")
    print(f"[INFO] empirical epsilon  : {emp_eps}")
    print(f"[INFO] empirical samples  : {emp_n}")

    L_emp = empirical_lipschitz_estimate(
        model=model,
        features_batch=sample_features,
        n_samples=emp_n,
        epsilon=emp_eps,
    )
    print(f"[RESULT] Empirical Lipschitz estimate (utility): {L_emp:.6g}")


    report = RobustnessSDP.decision_robustness_under_epsilon(
        model=model,
        sample_features=sample_features,
        epsilon=1000,
        lipschitz_bound=L_net,     # reuse computed bound; or set None to recompute
        p_structure="diag",
        solver="SCS",
        verbose=False,
        utility_mode="paid_minus_cost",
    )
    # attach empirical estimate
    report["empirical_lipschitz"] = {
        "epsilon": emp_eps,
        "n_samples": emp_n,
        "L_empirical": float(L_emp),
    }


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
