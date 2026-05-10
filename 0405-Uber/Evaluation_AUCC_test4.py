#!/usr/bin/env python
# coding: utf-8

"""
Non-RCT corrected AUCC evaluation.

本脚本保留你原本的两条主绘图链路：

1. strict_aucc_algorithm2
   - 对应原始 strict_aucc_algorithm2
   - 逐点排序、逐样本 IPS pseudo outcome、直接 cumsum
   - 不做 cost-bin 分桶
   - 保存 JSON 后由 plot_aucc_from_json 绘图
   - 输出：
       result/result_aucc_strict_nonrct.json
       result/aucc_curves_strict_nonrct_<timestamp>.pdf

2. calculate_and_save_aucc
   - 对应原始公司版 calculate_and_save_aucc
   - 先构造 corrected delta_gain / delta_cost
   - 恢复原始公司版 pd.cut(delta_cost) 分桶 + groupby().first() + sort_values(delta_cost)
   - 保存 JSON 后由 plot_aucc_from_json 绘图
   - 输出：
       result/result_aucc_costbin_nonrct.json
       result/aucc_curves_costbin_nonrct_<timestamp>.pdf

说明：
- get_aucc_plot 的 uplift/roi 两张即时 PNG 图在本版中去掉。
- 如果你的数据有真实 logged propensity，请直接提供 eval_df["propensity"]，不要重新估计。
- 如果没有真实 propensity，本脚本会使用 pre-treatment 数值特征估计 propensity，并显式过滤 treatment/paid/cost/uplift/roi 等泄漏字段。
"""

from __future__ import print_function, absolute_import, division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import io
import json
import random
import datetime
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except ImportError:
    LogisticRegression = None
    StandardScaler = None
    make_pipeline = None


# ==================== 路径与数据工具 ====================
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_utils_UScensus import *  # noqa


# ==================== 随机种子 ====================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"已设置随机种子: {seed}")


set_seeds(42)


# ==================== stdout/stderr 同步写日志 ====================
class TeeStream(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, s):
        if not s:
            return 0
        with self._lock:
            for st in self.streams:
                st.write(s)
            return len(s)

    def flush(self):
        with self._lock:
            for st in self.streams:
                st.flush()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    def isatty(self):
        return any(getattr(st, "isatty", lambda: False)() for st in self.streams)


@contextmanager
def tee_output(filepath, mode="a", encoding="utf-8"):
    old_out, old_err = sys.stdout, sys.stderr
    f = open(filepath, mode, encoding=encoding, buffering=1)
    try:
        sys.stdout = TeeStream(old_out, f)
        sys.stderr = TeeStream(old_err, f)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            f.close()


# ==================== 配置 ====================
config = {
    "eval_data": "../data/census1990_test.csv",
    "batch_size": 1024,
    "max_batches_for_eval": 79,

    # 两套 corrected AUCC JSON
    "aucc_strict_save_path": "result/result_aucc_strict_nonrct.json",
    "aucc_costbin_save_path": "result/result_aucc_costbin_nonrct.json",
}

model_paths_DFCL = [
    "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",
]

model_paths_else = []

model_name_map = {
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0": "CC-DFL",
}

treatment_order = [1, 0]
ratios = [i / 100.0 for i in range(5, 105, 5)]

aucc_strict_save_path = config["aucc_strict_save_path"]
aucc_costbin_save_path = config["aucc_costbin_save_path"]

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

Path("result").mkdir(parents=True, exist_ok=True)


# ==================== 数据加载 ====================
dataset = CSVData()
eval_samples = dataset.prepare_dataset(
    config["eval_data"],
    phase="eval",
    batch_size=config["batch_size"],
    shuffle=False,
)

label_name_list = ["treatment", "paid", "cost"]
drop_list = ["paid", "cost"]


def _to_features_labels(parsed_example):
    """
    保留你原来的拆分方式。
    注意：features 中可能含 treatment；后面收集 propensity 特征时会强制排除泄漏列。
    """
    features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}
    labels = {name: parsed_example[name] for name in label_name_list}
    return features, labels


eval_samples = eval_samples.map(
    _to_features_labels,
    num_parallel_calls=4,
).prefetch(1)


# ==================== Non-RCT correction helpers ====================
def _safe_np_1d(x, dtype=float):
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=dtype).reshape(-1)


def _clip_propensity(p, eps=0.02):
    p = _safe_np_1d(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def _is_leakage_feature_name(name: str) -> bool:
    """
    只要疑似 treatment/outcome/model prediction 泄漏，就不用于 propensity model。
    """
    name = str(name).lower()

    exact_bad = {
        "treatment",
        "_treatment_index",
        "paid",
        "cost",
        "label",
        "labels",
        "propensity",
        "logged_propensity",
    }
    if name in exact_bad:
        return True

    bad_keywords = [
        "treatment",
        "paid",
        "cost",
        "label",
        "outcome",
        "propensity",
        "uplift",
        "roi",
        "treat_",
        "ctrl_",
        "dr_",
        "ips_",
        "snips_",
    ]
    return any(k in name for k in bad_keywords)


def collect_pre_treatment_numeric_features(features_batch: Dict[str, Any]) -> pd.DataFrame:
    """
    只收集一维数值型 pre-treatment covariates，用于估计 propensity。
    排除 treatment / paid / cost / uplift / roi / prediction 等泄漏字段。
    """
    feature_batch_np = {}

    for fname, fval in features_batch.items():
        if _is_leakage_feature_name(fname):
            continue

        arr = fval.numpy() if hasattr(fval, "numpy") else np.asarray(fval)
        arr = np.asarray(arr)

        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            feature_batch_np[f"x__{fname}"] = arr

    return pd.DataFrame(feature_batch_np)


def estimate_propensity_from_feature_columns(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    feature_prefix: str = "x__",
    propensity_col: str = "propensity",
    eps: float = 0.02,
) -> pd.DataFrame:
    """
    使用 pre-treatment covariates 估计 e(x)=P(T=1|X)。

    如果数据有真实 logged propensity，请优先直接提供 df[propensity_col]，
    而不是使用这里的估计版本。
    """
    raw_feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]

    feature_cols = []
    dropped_cols = []
    for c in raw_feature_cols:
        c_without_prefix = c[len(feature_prefix):]
        if _is_leakage_feature_name(c_without_prefix) or _is_leakage_feature_name(c):
            dropped_cols.append(c)
        else:
            feature_cols.append(c)

    print(f"[Propensity] 原始候选特征数: {len(raw_feature_cols)}")
    print(f"[Propensity] 删除疑似泄漏特征数: {len(dropped_cols)}")
    if dropped_cols[:20]:
        print(f"[Propensity] 删除示例: {dropped_cols[:20]}")

    if len(feature_cols) == 0:
        raise ValueError(
            "没有可用于估计 propensity 的 pre-treatment 数值特征。"
            "请检查 features_batch，或直接提供真实 logged propensity 列。"
        )

    if LogisticRegression is None:
        raise ImportError(
            "需要 scikit-learn 来估计 propensity。请安装：pip install scikit-learn；"
            "或者直接提供真实 logged propensity 列。"
        )

    tmp = df[[treatment_col] + feature_cols].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = tmp[treatment_col].astype(int).values
    X = tmp[feature_cols].astype(float).values

    if len(np.unique(y)) < 2:
        raise ValueError("treatment 只有一个取值，无法估计 propensity。")

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        ),
    )
    clf.fit(X, y)

    p = clf.predict_proba(X)[:, 1]
    p = _clip_propensity(p, eps=eps)
    df[propensity_col] = p

    print(
        f"[Propensity] 使用 {len(feature_cols)} 个 pre-treatment 特征估计完成："
        f"min={np.min(p):.4f}, "
        f"p01={np.quantile(p, 0.01):.4f}, "
        f"mean={np.mean(p):.4f}, "
        f"p99={np.quantile(p, 0.99):.4f}, "
        f"max={np.max(p):.4f}"
    )

    return df


def _prepare_nonrct_eval_df(
    df: pd.DataFrame,
    treatment_col: str,
    treatment_val: int,
    control_val: int,
    propensity_col: str,
    eps: float,
    use_estimated_propensity_if_missing: bool,
    feature_prefix: str,
) -> pd.DataFrame:
    """
    筛选 treatment/control，并准备 propensity。
    """
    df_filtered = df[
        (df[treatment_col] == control_val) | (df[treatment_col] == treatment_val)
    ].copy()
    df_filtered[treatment_col] = (df_filtered[treatment_col] == treatment_val).astype(int)

    if propensity_col not in df_filtered.columns:
        if not use_estimated_propensity_if_missing:
            raise ValueError(
                f"缺少 {propensity_col!r}。非 RCT 评估必须提供 logged propensity，"
                "或者允许用 pre-treatment covariates 估计 propensity。"
            )
        df_filtered = estimate_propensity_from_feature_columns(
            df_filtered,
            treatment_col=treatment_col,
            feature_prefix=feature_prefix,
            propensity_col=propensity_col,
            eps=eps,
        )
    else:
        df_filtered[propensity_col] = _clip_propensity(df_filtered[propensity_col].values, eps=eps)

    return df_filtered


def _save_curve_json(
    save_path: str,
    model_key: str,
    aucc_score: float,
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    meta: Dict[str, Any],
    propensity_col: str,
    eps: float,
):
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    all_results[model_key] = {
        "aucc_score": float(aucc_score),
        "x_coords": x.tolist(),
        "y_coords": y.tolist(),
        "method": method,
        "x_axis": meta.get("x_axis"),
        "x_label": meta.get("x_label"),
        "propensity_col": propensity_col,
        "eps": eps,
        "final_gain": meta.get("final_gain"),
        "final_incremental_cost": meta.get("final_incremental_cost"),
        "final_policy_cost": meta.get("final_policy_cost"),
        "x_min": float(np.nanmin(x)) if len(x) else None,
        "x_max": float(np.nanmax(x)) if len(x) else None,
        "y_min": float(np.nanmin(y)) if len(y) else None,
        "y_max": float(np.nanmax(y)) if len(y) else None,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)


# ==================== 两个保留函数：均已改成 Non-RCT corrected ====================
def strict_aucc_algorithm2(
    df: pd.DataFrame,
    reward_col: str = "paid",
    cost_col: str = "cost",
    treatment_col: str = "treatment",
    uplift_col: str = "uplift",
    propensity_col: str = "propensity",
    treatment_val: int = 1,
    control_val: int = 0,
    model_key: Optional[str] = None,
    save_path: Optional[str] = None,
    eps: float = 0.02,
    use_estimated_propensity_if_missing: bool = True,
    feature_prefix: str = "x__",
    x_axis: str = "policy_cost",
    save_debug_csv_path: Optional[str] = None,
) -> float:
    """
    矫正后的 strict_aucc_algorithm2。

    保持原始 strict 语义：
    - 按 uplift_col 降序排序；
    - 不分桶；
    - 不按 delta_cost 重新排序；
    - 每个样本先做 IPS pseudo outcome，再逐点 cumsum；
    - 保存 JSON 后由 plot_aucc_from_json 统一绘图。

    x_axis:
    - "policy_cost": 推荐；横轴通常单调稳定，图更像标准 cost curve。
    - "incremental_cost": 更接近原始 strict 的 delta_C 横轴，但 corrected delta_C 可能非单调。
    - "fraction": 用 selected fraction 做横轴，便于排查排序质量。
    """
    if save_path is None:
        save_path = aucc_strict_save_path
    if model_key is None:
        model_key = "unknown_model"

    df_filtered = _prepare_nonrct_eval_df(
        df=df,
        treatment_col=treatment_col,
        treatment_val=treatment_val,
        control_val=control_val,
        propensity_col=propensity_col,
        eps=eps,
        use_estimated_propensity_if_missing=use_estimated_propensity_if_missing,
        feature_prefix=feature_prefix,
    )

    df_sorted = df_filtered.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        raise ValueError("strict_aucc_algorithm2 输入为空。")

    t = df_sorted[treatment_col].astype(int).values
    e = _clip_propensity(df_sorted[propensity_col].values, eps=eps)

    reward = df_sorted[reward_col].astype(float).values
    cost = df_sorted[cost_col].astype(float).values

    # 逐样本 IPS pseudo outcome：corrected ΔR_i / ΔC_i
    ips_delta_reward = t * reward / e - (1 - t) * reward / (1 - e)
    ips_delta_cost = t * cost / e - (1 - t) * cost / (1 - e)

    # strict：逐点累计，不做 bin，不重排
    delta_R_list = np.cumsum(ips_delta_reward)
    delta_C_list = np.cumsum(ips_delta_cost)

    # treatment-side policy cost，通常非负单调
    policy_cost = np.cumsum(t * cost / e)
    selected_fraction = np.arange(1, n + 1, dtype=float) / n

    if x_axis == "policy_cost":
        x_raw = np.r_[0.0, policy_cost]
        x_label = "Normalized Policy Cost"
    elif x_axis == "fraction":
        x_raw = np.r_[0.0, selected_fraction]
        x_label = "Selected Fraction"
    elif x_axis == "incremental_cost":
        x_raw = np.r_[0.0, delta_C_list]
        x_label = "Normalized Incremental Cost"
    else:
        raise ValueError("x_axis 只能是 'policy_cost', 'fraction', 'incremental_cost'。")

    y_raw = np.r_[0.0, delta_R_list]

    x_den = x_raw[-1]
    y_den = y_raw[-1]

    if np.isclose(x_den, 0.0):
        print("[strict corrected] x 终点接近 0，横轴回退为 selected_fraction。")
        x = np.r_[0.0, selected_fraction]
        x_label = "Selected Fraction"
        x_axis_used = "fraction"
    else:
        x = x_raw / x_den
        x_axis_used = x_axis

    if np.isclose(y_den, 0.0):
        print("[strict corrected] y 终点接近 0，改用 max(abs(y)) 归一化。")
        alt_den = np.nanmax(np.abs(y_raw))
        y = y_raw if np.isclose(alt_den, 0.0) else y_raw / alt_den
    else:
        y = y_raw / y_den

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # strict 不做 pd.cut、不做 cost-bin、不重排 delta_cost。
    # 对 policy_cost/fraction，仅去重并按 x 排序，保证 PDF 绘图不会来回连线。
    if x_axis_used in ("policy_cost", "fraction"):
        tmp = pd.DataFrame({"x": x, "y": y})
        tmp = tmp.groupby("x", as_index=False)["y"].last().sort_values("x")
        x = tmp["x"].values
        y = tmp["y"].values

    aucc_score = float(np.trapz(y, x))

    if save_debug_csv_path:
        debug_df = df_sorted.copy()
        debug_df["ips_delta_reward"] = ips_delta_reward
        debug_df["ips_delta_cost"] = ips_delta_cost
        debug_df["cum_ips_delta_reward"] = delta_R_list
        debug_df["cum_ips_delta_cost"] = delta_C_list
        debug_df["ips_policy_cost"] = policy_cost
        debug_df["selected_fraction"] = selected_fraction
        debug_df.to_csv(save_debug_csv_path, index=False, encoding="utf-8-sig")
        print(f"[strict corrected] debug 已保存: {save_debug_csv_path}")

    meta = {
        "x_axis": x_axis_used,
        "x_label": x_label,
        "final_gain": float(y_raw[-1]),
        "final_incremental_cost": float(delta_C_list[-1]),
        "final_policy_cost": float(policy_cost[-1]),
    }

    _save_curve_json(
        save_path=save_path,
        model_key=model_key,
        aucc_score=aucc_score,
        x=x,
        y=y,
        method="strict_aucc_algorithm2_nonrct_ips_cumsum",
        meta=meta,
        propensity_col=propensity_col,
        eps=eps,
    )

    print(
        f"[strict corrected] 保存成功: {save_path}, "
        f"score={aucc_score:.6f}, "
        f"x_axis={x_axis_used}, "
        f"x_range=({np.nanmin(x):.4f}, {np.nanmax(x):.4f}), "
        f"y_range=({np.nanmin(y):.4f}, {np.nanmax(y):.4f}), "
        f"final_gain={meta['final_gain']:.6f}, "
        f"final_incremental_cost={meta['final_incremental_cost']:.6f}, "
        f"final_policy_cost={meta['final_policy_cost']:.6f}"
    )

    return aucc_score


def calculate_and_save_aucc(
    df: pd.DataFrame,
    reward_col: str = "paid",
    cost_col: str = "cost",
    treatment_col: str = "treatment",
    uplift_col: str = "uplift",
    propensity_col: str = "propensity",
    treatment_val: int = 1,
    control_val: int = 0,
    n_bins: int = 100,
    model_key: Optional[str] = None,
    save_path: Optional[str] = None,
    eps: float = 0.02,
    use_estimated_propensity_if_missing: bool = True,
    feature_prefix: str = "x__",
    output_dip_samples_path: Optional[str] = None,
    save_debug_csv_path: Optional[str] = None,
) -> float:
    """
    矫正后的 calculate_and_save_aucc。

    保持原始公司版语义：
    - 按 uplift_col 排序；
    - 先构造 corrected delta_gain / delta_cost；
    - 按 delta_cost 做 pd.cut 分桶；
    - 每桶取第一个点；
    - append 最后一个点；
    - 按 delta_cost 排序后保存坐标；
    - 这张图允许出现 cost-bin 带来的跳变/抖动。
    """
    if save_path is None:
        save_path = aucc_costbin_save_path
    if model_key is None:
        model_key = "unknown_model"

    df_filtered = _prepare_nonrct_eval_df(
        df=df,
        treatment_col=treatment_col,
        treatment_val=treatment_val,
        control_val=control_val,
        propensity_col=propensity_col,
        eps=eps,
        use_estimated_propensity_if_missing=use_estimated_propensity_if_missing,
        feature_prefix=feature_prefix,
    )

    df_sorted = df_filtered.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        raise ValueError("calculate_and_save_aucc 输入为空。")

    # 与原始公司版一致：index 从 1 开始
    df_sorted.index = df_sorted.index + 1

    t = df_sorted[treatment_col].astype(int).values
    e = _clip_propensity(df_sorted[propensity_col].values, eps=eps)

    reward = df_sorted[reward_col].astype(float).values
    cost = df_sorted[cost_col].astype(float).values

    ips_delta_reward = t * reward / e - (1 - t) * reward / (1 - e)
    ips_delta_cost = t * cost / e - (1 - t) * cost / (1 - e)

    # corrected cumulative uplift
    df_sorted["delta_gain"] = np.cumsum(ips_delta_reward)
    df_sorted["delta_cost"] = np.cumsum(ips_delta_cost)
    df_sorted["policy_cost"] = np.cumsum(t * cost / e)
    df_sorted["selected_fraction"] = np.arange(1, n + 1, dtype=float) / n

    # 恢复原始公司版分桶逻辑：按 delta_cost 分桶
    try:
        df_sorted["cost_bin"] = pd.cut(
            df_sorted["delta_cost"],
            bins=n_bins,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        )
    except ValueError:
        print("[calculate corrected] delta_cost 无法分桶，回退为按排序位置取点。")
        idx = np.unique(np.ceil(np.linspace(1, n, min(n_bins, n))).astype(int) - 1)
        df_binned = df_sorted.iloc[idx].copy().reset_index()
    else:
        df_binned = df_sorted.dropna(subset=["cost_bin"]).groupby("cost_bin").first()

        # 确保曲线终点是全体用户
        last_row = df_sorted.iloc[[-1]]
        df_binned = pd.concat([df_binned, last_row]).reset_index()
        df_binned = (
            df_binned
            .drop_duplicates(subset=["delta_cost"], keep="first")
            .sort_values("delta_cost")
            .reset_index(drop=True)
        )

    if len(df_binned) == 0:
        raise ValueError("calculate_and_save_aucc 分桶后为空。")

    final_delta_gain = float(df_binned["delta_gain"].iloc[-1])
    final_delta_cost = float(df_binned["delta_cost"].iloc[-1])

    if np.isclose(final_delta_gain, 0.0):
        y_den = np.nanmax(np.abs(df_binned["delta_gain"].values))
        if np.isclose(y_den, 0.0):
            y = np.zeros(len(df_binned) + 1)
        else:
            y = np.r_[0.0, df_binned["delta_gain"].values / y_den]
    else:
        y = np.r_[0.0, df_binned["delta_gain"].values / final_delta_gain]

    if np.isclose(final_delta_cost, 0.0):
        print("[calculate corrected] final_delta_cost 接近 0，横轴回退为 policy_cost。")
        x_raw = np.r_[0.0, df_binned["policy_cost"].values]
        x_den = x_raw[-1]
        if np.isclose(x_den, 0.0):
            x = np.r_[0.0, df_binned["selected_fraction"].values]
            x_axis_used = "fraction"
            x_label = "Selected Fraction"
        else:
            x = x_raw / x_den
            x_axis_used = "policy_cost"
            x_label = "Normalized Policy Cost"
    else:
        # 公司版保留 delta_cost 横轴
        x = np.r_[0.0, df_binned["delta_cost"].values / final_delta_cost]
        x_axis_used = "incremental_cost_binned"
        x_label = "Normalized Incremental Cost"

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # 注意：公司版保留按 delta_cost 排序后的路径，不再额外平滑。
    aucc_score = float(np.trapz(y, x))

    if output_dip_samples_path and len(df_binned) > 1:
        dip_rows = []
        for i in range(len(df_binned) - 1):
            if df_binned.loc[i + 1, "delta_gain"] < df_binned.loc[i, "delta_gain"]:
                dip_rows.append(df_binned.iloc[[i, i + 1]])
        if dip_rows:
            pd.concat(dip_rows).to_csv(output_dip_samples_path, index=False, encoding="utf-8-sig")
            print(f"[calculate corrected] 下降区间已保存: {output_dip_samples_path}")

    if save_debug_csv_path:
        df_sorted.to_csv(save_debug_csv_path, index=False, encoding="utf-8-sig")
        print(f"[calculate corrected] debug 已保存: {save_debug_csv_path}")

    meta = {
        "x_axis": x_axis_used,
        "x_label": x_label,
        "final_gain": final_delta_gain,
        "final_incremental_cost": final_delta_cost,
        "final_policy_cost": float(df_binned["policy_cost"].iloc[-1]),
    }

    _save_curve_json(
        save_path=save_path,
        model_key=model_key,
        aucc_score=aucc_score,
        x=x,
        y=y,
        method="calculate_and_save_aucc_nonrct_ips_costbin",
        meta=meta,
        propensity_col=propensity_col,
        eps=eps,
    )

    print(
        f"[calculate corrected] 保存成功: {save_path}, "
        f"score={aucc_score:.6f}, "
        f"x_axis={x_axis_used}, "
        f"x_range=({np.nanmin(x):.4f}, {np.nanmax(x):.4f}), "
        f"y_range=({np.nanmin(y):.4f}, {np.nanmax(y):.4f}), "
        f"final_gain={final_delta_gain:.6f}, "
        f"final_incremental_cost={final_delta_cost:.6f}"
    )

    return aucc_score


# ==================== 从 JSON 绘图 ====================
def plot_aucc_from_json(
    json_path: str,
    plot_path: str = "aucc_comparison.pdf",
    model_names: Optional[List[str]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    fallback_to_basename: bool = True,
):
    legend_fs = 16
    axis_label_fs = legend_fs
    tick_fs = legend_fs

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_results: Dict[str, Dict[str, Any]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取或解析文件 {json_path} 时出错: {e}")
        return

    if not all_results:
        print("JSON 文件为空或格式不正确，无法绘图。")
        return

    model_name_map = model_name_map or {}

    if model_names:
        ordered_keys = [k for k in model_names if k in all_results]
        not_found = [k for k in model_names if k not in all_results]
        if not_found:
            print(f"警告: 在 {json_path} 中未找到以下模型: {not_found}")
    else:
        ordered_keys = list(all_results.keys())

    if not ordered_keys:
        print("没有找到可供绘制的模型数据。")
        return

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update({
        "font.size": legend_fs,
        "axes.labelsize": axis_label_fs,
        "xtick.labelsize": tick_fs,
        "ytick.labelsize": tick_fs,
        "legend.fontsize": legend_fs,
    })

    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    color_idx = 0
    base_lw = 2.5
    highlight_name = "CC-DFL"

    plt.figure(figsize=(7.2, 5.4))
    all_y = []
    x_label = None
    any_incremental_cost = False

    for model_key in ordered_keys:
        data = all_results.get(model_key, {})
        if not ("x_coords" in data and "y_coords" in data and "aucc_score" in data):
            print(f"模型 '{model_key}' 的数据不完整，跳过绘图。")
            continue

        display_name = model_name_map.get(model_key)
        if not display_name:
            display_name = Path(model_key).name if fallback_to_basename else model_key

        is_highlight = display_name == highlight_name
        if is_highlight:
            color = "red"
            lw = base_lw * 1.2
            zorder = 10
        else:
            color = colors[color_idx % len(colors)]
            color_idx += 1
            lw = base_lw
            zorder = 2

        x = np.asarray(data["x_coords"], dtype=float)
        y = np.asarray(data["y_coords"], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        x_axis = data.get("x_axis", "policy_cost")
        if x_axis in ("fraction", "policy_cost"):
            order = np.argsort(x)
            x = x[order]
            y = y[order]
        else:
            any_incremental_cost = True

        method = data.get("method", "nonrct_corrected")
        x_label = data.get("x_label", x_label or "Normalized Policy Cost")
        all_y.extend(y.tolist())

        score = data["aucc_score"]
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) and np.isfinite(score) else str(score)

        plt.plot(
            x,
            y,
            linestyle="-",
            linewidth=lw,
            color=color,
            marker=None,
            zorder=zorder,
            label=f"{display_name} {method} ({score_text})",
        )

    # random baseline：只有在横轴大致归一化 [0,1] 时才有清晰参考意义
    plt.plot(
        [0, 1],
        [0, 1],
        color="k",
        linestyle="--",
        linewidth=base_lw,
        label="Random",
        alpha=0.3,
    )

    # policy_cost/fraction 固定到 [0,1]；incremental_cost_binned 可能略出界，不强制 xlim
    if not any_incremental_cost:
        plt.xlim(0, 1)

    if len(all_y) > 0:
        ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
        margin = 0.05 * max(1.0, ymax - ymin)
        plt.ylim(ymin - margin, ymax + margin)

    plt.xlabel(x_label or "Normalized Policy Cost", fontsize=axis_label_fs)
    plt.ylabel("Normalized Incremental Reward", fontsize=axis_label_fs)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.legend(fontsize=legend_fs, frameon=True)
    plt.grid(True, alpha=0.3)

    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"AUCC 曲线对比图已保存至: {plot_path}")


# ==================== 评估主流程 ====================
print("开始评估流程...")

for model_path in model_paths_DFCL:
    print(f"\n{'=' * 20} 正在评估模型: {model_path} {'=' * 20}")

    result_dir = Path(f"./{model_path}/result")
    result_dir.mkdir(parents=True, exist_ok=True)
    print("result文件夹已创建...")

    print("加载模型并进行预测...")
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = tf.keras.models.load_model(model_path, compile=False)

    all_uplifts = []
    all_uplift_paid = []
    all_uplift_cost = []
    all_rois = []
    all_treat_paid = []
    all_treat_cost = []
    all_ctrl_paid = []
    all_ctrl_cost = []
    all_paid_labels = []
    all_cost_labels = []
    all_treatment_labels = []
    all_feature_batches = []

    print("开始分批次进行预测...")
    max_batches_for_eval = config["max_batches_for_eval"]

    for i, (features_batch, labels_batch) in enumerate(eval_samples):
        if max_batches_for_eval is not None and i >= max_batches_for_eval:
            print(f"已达到最大评估批次数 {max_batches_for_eval}，停止预测。")
            break

        predictions_logits = model.predict(features_batch)

        pred_dict = {
            key: tf.exp(tf.minimum(logit, 10.0))
            for key, logit in predictions_logits.items()
        }

        pred_paid_treat = pred_dict["paid_treatment_1"]
        pred_cost_treat = pred_dict["cost_treatment_1"]
        pred_paid_ctrl = pred_dict["paid_treatment_0"]
        pred_cost_ctrl = pred_dict["cost_treatment_0"]

        num_samples = len(pred_paid_treat)
        total_uplift_per_sample = np.zeros(num_samples)

        for r in ratios:
            uplift = pred_paid_treat - r * pred_cost_treat
            total_uplift_per_sample += uplift

        integrated_uplift_per_sample = total_uplift_per_sample / len(ratios)

        pred_paid_uplift = pred_paid_treat - pred_paid_ctrl
        pred_cost_uplift = pred_cost_treat - pred_cost_ctrl

        roi_tensor = tf.where(
            pred_cost_uplift > 0,
            tf.math.divide_no_nan(pred_paid_uplift, pred_cost_uplift),
            tf.zeros_like(pred_paid_uplift),
        )
        roi = roi_tensor.numpy()

        all_uplifts.append(_safe_np_1d(integrated_uplift_per_sample))
        all_uplift_paid.append(_safe_np_1d(pred_paid_uplift))
        all_uplift_cost.append(_safe_np_1d(pred_cost_uplift))
        all_rois.append(_safe_np_1d(roi))
        all_treat_paid.append(_safe_np_1d(pred_paid_treat))
        all_treat_cost.append(_safe_np_1d(pred_cost_treat))
        all_ctrl_paid.append(_safe_np_1d(pred_paid_ctrl))
        all_ctrl_cost.append(_safe_np_1d(pred_cost_ctrl))
        all_paid_labels.append(_safe_np_1d(labels_batch["paid"]))
        all_cost_labels.append(_safe_np_1d(labels_batch["cost"]))
        all_treatment_labels.append(_safe_np_1d(labels_batch["treatment"], dtype=int))

        all_feature_batches.append(collect_pre_treatment_numeric_features(features_batch))

    print("所有批次预测完成，正在整合结果...")

    final_uplifts = np.concatenate(all_uplifts)
    final_uplift_paid = np.concatenate(all_uplift_paid)
    final_uplift_cost = np.concatenate(all_uplift_cost)
    final_rois = np.concatenate(all_rois)
    final_treat_paid = np.concatenate(all_treat_paid)
    final_treat_cost = np.concatenate(all_treat_cost)
    final_ctrl_paid = np.concatenate(all_ctrl_paid)
    final_ctrl_cost = np.concatenate(all_ctrl_cost)
    final_paid = np.concatenate(all_paid_labels)
    final_cost = np.concatenate(all_cost_labels)
    final_treatment = np.concatenate(all_treatment_labels)

    print("正在将所有结果整合到 DataFrame...")
    eval_df = pd.DataFrame({
        "paid": final_paid,
        "cost": final_cost,
        "treatment": final_treatment,
        "uplift": final_uplifts,
        "roi": final_rois,
        "treat_paid": final_treat_paid,
        "treat_cost": final_treat_cost,
        "ctrl_paid": final_ctrl_paid,
        "ctrl_cost": final_ctrl_cost,
        "uplift_paid": final_uplift_paid,
        "uplift_cost": final_uplift_cost,
    })

    if len(all_feature_batches) > 0:
        feature_df = pd.concat(all_feature_batches, axis=0, ignore_index=True)
        feature_df = feature_df.reset_index(drop=True)

        if len(feature_df) != len(eval_df):
            raise ValueError(
                f"feature_df 行数 {len(feature_df)} 与 eval_df 行数 {len(eval_df)} 不一致。"
            )

        eval_df = pd.concat([eval_df.reset_index(drop=True), feature_df], axis=1)
        print(f"[Non-RCT] 已拼接 pre-treatment propensity 特征列数量: {feature_df.shape[1]}")

    with tee_output(f"{model_path}/eval.log", mode="a", encoding="utf-8"):
        print("\n评估结果 DataFrame 示例:")
        print(eval_df.head())
        eval_df["treatment"] = eval_df["treatment"].astype(int)

        # 如果有真实 logged propensity，请在这里打开并替换列名：
        # eval_df["propensity"] = eval_df["logged_propensity"]

        print("正在计算 corrected strict_aucc_algorithm2...")
        aucc_score_strict = strict_aucc_algorithm2(
            df=eval_df,
            reward_col="paid",
            cost_col="cost",
            treatment_col="treatment",
            uplift_col="uplift",
            propensity_col="propensity",
            treatment_val=1,
            control_val=0,
            model_key=model_path,
            save_path=aucc_strict_save_path,
            eps=0.02,
            use_estimated_propensity_if_missing=True,
            feature_prefix="x__",
            x_axis="policy_cost",
            save_debug_csv_path=f"{model_path}/result/aucc_strict_nonrct_debug.csv",
        )
        print(f"模型 {model_path} corrected strict AUCC 分数为: {aucc_score_strict:.6f}")

        print("正在计算 corrected calculate_and_save_aucc...")
        aucc_score_costbin = calculate_and_save_aucc(
            df=eval_df,
            reward_col="paid",
            cost_col="cost",
            treatment_col="treatment",
            uplift_col="uplift",
            propensity_col="propensity",
            treatment_val=1,
            control_val=0,
            n_bins=100,
            model_key=model_path,
            save_path=aucc_costbin_save_path,
            eps=0.02,
            use_estimated_propensity_if_missing=True,
            feature_prefix="x__",
            output_dip_samples_path=f"{model_path}/result/aucc_costbin_nonrct_dip_samples.csv",
            save_debug_csv_path=f"{model_path}/result/aucc_costbin_nonrct_debug.csv",
        )
        print(f"模型 {model_path} corrected cost-bin AUCC 分数为: {aucc_score_costbin:.6f}")

        # get_aucc_plot 的 uplift/roi 两张图在新评估中去掉。


# ==================== 两次 plot_aucc_from_json：对应两张最终图 ====================
strict_output_image_path = f"result/aucc_curves_strict_nonrct_{current_time}.pdf"
plot_aucc_from_json(
    aucc_strict_save_path,
    strict_output_image_path,
    model_names=model_paths_DFCL + model_paths_else,
    model_name_map=model_name_map,
)

costbin_output_image_path = f"result/aucc_curves_costbin_nonrct_{current_time}.pdf"
plot_aucc_from_json(
    aucc_costbin_save_path,
    costbin_output_image_path,
    model_names=model_paths_DFCL + model_paths_else,
    model_name_map=model_name_map,
)
