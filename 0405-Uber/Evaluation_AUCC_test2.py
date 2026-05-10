#!/usr/bin/env python
# coding: utf-8

"""
Non-RCT AUCC Evaluation Script

核心改动：
1. 对非 RCT 数据，不再用 treated/control 原始差分直接评估 AUCC。
2. 使用 propensity score + IPS/SNIPS 校正 offline evaluation。
3. propensity 只允许使用 pre-treatment covariates，显式排除 treatment/paid/cost/uplift/roi 等泄漏字段。
4. AUCC 横轴默认使用 policy_cost，而不是 incremental_cost，避免 final incremental cost 接近 0 时曲线挤到 x=0。
5. 输出 debug csv，便于检查 propensity overlap、SNIPS 累计收益/成本曲线。
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
    "aucc_save_path": "result/result_aucc_nonrct_ips_snips.json",
    "auuc_save_path": "result/result_auuc.json",
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
aucc_save_path = config["aucc_save_path"]
auuc_save_path = config["auuc_save_path"]
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
    注意：这里保留原始写法，features 中可能仍含 treatment。
    后面收集 propensity 特征时会再次强制排除 treatment/paid/cost 等泄漏列。
    """
    features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}
    labels = {}
    for name in label_name_list:
        labels[name] = parsed_example[name]
    return features, labels


eval_samples = eval_samples.map(
    _to_features_labels,
    num_parallel_calls=4,
).prefetch(1)


# ==================== Non-RCT AUCC 工具函数 ====================
def _safe_np_1d(x, dtype=float):
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(-1)


def _clip_propensity(p, eps=0.02):
    p = _safe_np_1d(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def _is_leakage_feature_name(name: str) -> bool:
    """
    判断特征名是否可能泄漏 treatment/outcome/model prediction。
    只要疑似泄漏，就不要用于 propensity model。
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
        "snips_",
        "ips_",
    ]
    return any(k in name for k in bad_keywords)


def collect_pre_treatment_numeric_features(features_batch: Dict[str, Any]) -> pd.DataFrame:
    """
    从 features_batch 中收集一维数值型 pre-treatment covariates。
    明确排除 treatment/paid/cost/uplift/roi/propensity 等泄漏字段。
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

    强烈建议：
    - 如果数据有真实 logged propensity，直接提供 df[propensity_col]，不要估计。
    - 如果必须估计，严禁把 treatment/outcome/model prediction 放入 feature_cols。
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


def calculate_and_save_aucc_nonrct_ips_snips(
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
    n_bins: int = 100,
    use_estimated_propensity_if_missing: bool = True,
    feature_prefix: str = "x__",
    x_axis: str = "policy_cost",
    save_debug_csv_path: Optional[str] = None,
) -> float:
    """
    Non-RCT corrected AUCC using IPS/SNIPS.

    只用模型输出 uplift_col 做排序；
    用 propensity 对 observed reward/cost 做校正；
    不使用当前被评估模型自己的 treat_paid/ctrl_paid 参与评估，避免“模型自评估”。

    x_axis:
        'fraction':
            横轴 = selected population fraction，最稳定，适合排查排序质量。
        'policy_cost':
            横轴 = top-k policy 的累计 treatment-side cost，通常非负且单调，推荐作为 cost curve 横轴。
        'incremental_cost':
            横轴 = SNIPS incremental cost，即 C1-C0；不保证单调，不推荐作为默认画图横轴。
    """
    if save_path is None:
        save_path = aucc_save_path
    if model_key is None:
        model_key = "unknown_model"

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
        df_filtered[propensity_col] = _clip_propensity(
            df_filtered[propensity_col].values,
            eps=eps,
        )

    df_sorted = df_filtered.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        raise ValueError("AUCC 输入为空。")

    t = df_sorted[treatment_col].astype(int).values
    e = _clip_propensity(df_sorted[propensity_col].values, eps=eps)
    y = df_sorted[reward_col].astype(float).values
    c = df_sorted[cost_col].astype(float).values

    # IPS weights
    w_t = t / e
    w_c = (1 - t) / (1 - e)

    # SNIPS prefix means for top-k
    den_t = np.cumsum(w_t)
    den_c = np.cumsum(w_c)

    sum_y_t = np.cumsum(w_t * y)
    sum_y_c = np.cumsum(w_c * y)

    sum_c_t = np.cumsum(w_t * c)
    sum_c_c = np.cumsum(w_c * c)

    eps_den = 1e-12
    mean_y_t = sum_y_t / np.maximum(den_t, eps_den)
    mean_y_c = sum_y_c / np.maximum(den_c, eps_den)

    mean_c_t = sum_c_t / np.maximum(den_t, eps_den)
    mean_c_c = sum_c_c / np.maximum(den_c, eps_den)

    k_arr = np.arange(1, n + 1, dtype=float)

    # Corrected top-k incremental reward / cost
    delta_gain = (mean_y_t - mean_y_c) * k_arr
    delta_cost = (mean_c_t - mean_c_c) * k_arr

    # Treatment-side policy cost, usually non-negative and monotone
    policy_cost = sum_c_t
    selected_fraction = k_arr / n

    # 按排序位置分桶，不按 cost 分桶
    if n_bins is not None and n_bins > 0 and n > n_bins:
        idx = np.unique(np.ceil(np.linspace(1, n, n_bins)).astype(int) - 1)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
    else:
        idx = np.arange(n)

    gain_plot = delta_gain[idx]
    incr_cost_plot = delta_cost[idx]
    policy_cost_plot = policy_cost[idx]
    frac_plot = selected_fraction[idx]

    # 加原点
    gain_plot = np.r_[0.0, gain_plot]
    incr_cost_plot = np.r_[0.0, incr_cost_plot]
    policy_cost_plot = np.r_[0.0, policy_cost_plot]
    frac_plot = np.r_[0.0, frac_plot]

    if x_axis == "fraction":
        x_raw = frac_plot
        x_label = "Selected Fraction"
    elif x_axis == "policy_cost":
        x_raw = policy_cost_plot
        x_label = "Normalized Policy Cost"
    elif x_axis == "incremental_cost":
        x_raw = incr_cost_plot
        x_label = "Normalized Incremental Cost"
    else:
        raise ValueError("x_axis 只能是 'fraction', 'policy_cost', 'incremental_cost'。")

    # x 归一化
    x_den = x_raw[-1]
    if np.isclose(x_den, 0.0):
        print(f"[AUCC-IPS] {x_axis} 终点接近 0，横轴改用 selected_fraction。")
        x = frac_plot
        x_label = "Selected Fraction"
        x_axis_used = "fraction"
    else:
        x = x_raw / x_den
        x_axis_used = x_axis

    # y 归一化
    y_den = gain_plot[-1]
    if np.isclose(y_den, 0.0):
        print("[AUCC-IPS] final incremental gain 接近 0，改用 max(abs(gain)) 做绘图尺度。")
        alt_den = np.nanmax(np.abs(gain_plot))
        y = gain_plot if np.isclose(alt_den, 0.0) else gain_plot / alt_den
    else:
        y = gain_plot / y_den

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # 对 fraction / policy_cost，横轴应单调；去重后积分
    if x_axis_used in ("fraction", "policy_cost"):
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        tmp = pd.DataFrame({"x": x, "y": y})
        tmp = tmp.groupby("x", as_index=False)["y"].last().sort_values("x")
        x = tmp["x"].values
        y = tmp["y"].values

    if np.any(np.diff(x) < -1e-12):
        print(
            f"[AUCC-IPS] 警告：x_axis={x_axis_used} 横轴非单调，"
            "trapz 面积可能不适合作为标准 AUCC。"
        )

    aucc_score = float(np.trapz(y, x))

    if save_debug_csv_path:
        debug_df = df_sorted.copy()
        debug_df["ips_w_t"] = w_t
        debug_df["ips_w_c"] = w_c
        debug_df["snips_delta_gain"] = delta_gain
        debug_df["snips_incremental_cost"] = delta_cost
        debug_df["ips_policy_cost"] = policy_cost
        debug_df["selected_fraction"] = selected_fraction
        debug_df.to_csv(save_debug_csv_path, index=False, encoding="utf-8-sig")
        print(f"[AUCC-IPS] debug 明细已保存: {save_debug_csv_path}")

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    all_results[model_key] = {
        "aucc_score": aucc_score,
        "x_coords": x.tolist(),
        "y_coords": y.tolist(),
        "method": "nonrct_ips_snips",
        "x_axis": x_axis_used,
        "x_label": x_label,
        "propensity_col": propensity_col,
        "eps": eps,
        "final_gain": float(gain_plot[-1]),
        "final_incremental_cost": float(incr_cost_plot[-1]),
        "final_policy_cost": float(policy_cost_plot[-1]),
        "x_min": float(np.nanmin(x)) if len(x) else None,
        "x_max": float(np.nanmax(x)) if len(x) else None,
        "y_min": float(np.nanmin(y)) if len(y) else None,
        "y_max": float(np.nanmax(y)) if len(y) else None,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(
        f"[AUCC-IPS] 保存成功: {save_path}, "
        f"score={aucc_score:.6f}, "
        f"x_axis={x_axis_used}, "
        f"x_range=({np.nanmin(x):.4f}, {np.nanmax(x):.4f}), "
        f"y_range=({np.nanmin(y):.4f}, {np.nanmax(y):.4f}), "
        f"final_gain={gain_plot[-1]:.6f}, "
        f"final_incremental_cost={incr_cost_plot[-1]:.6f}, "
        f"final_policy_cost={policy_cost_plot[-1]:.6f}"
    )

    return aucc_score


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

        # 对 policy_cost / fraction，x 应在 [0,1] 且单调；如果 JSON 已排序则不改变。
        # 对 incremental_cost，不建议使用这个绘图作为标准 AUCC。
        if data.get("x_axis", "policy_cost") in ("fraction", "policy_cost"):
            order = np.argsort(x)
            x = x[order]
            y = y[order]

        method = data.get("method", "nonrct_ips_snips")
        x_label = data.get("x_label", x_label or "Normalized Policy Cost")
        all_y.extend(y.tolist())

        plt.plot(
            x,
            y,
            linestyle="-",
            linewidth=lw,
            color=color,
            marker=None,
            zorder=zorder,
            label=f'{display_name} {method} ({data["aucc_score"]:.4f})',
        )

    # random baseline：在归一化横轴下画 y=x
    plt.plot(
        [0, 1],
        [0, 1],
        color="k",
        linestyle="--",
        linewidth=base_lw,
        label="Random",
        alpha=0.3,
    )

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

        # 只收集 pre-treatment 数值特征，避免 treatment/outcome 泄漏
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

        print("正在计算 Non-RCT IPS/SNIPS corrected AUCC 指标...")

        # 如果有真实 logged propensity，请在这里打开并替换列名：
        # eval_df["propensity"] = eval_df["logged_propensity"]

        aucc_score_ips = calculate_and_save_aucc_nonrct_ips_snips(
            df=eval_df,
            reward_col="paid",
            cost_col="cost",
            treatment_col="treatment",
            uplift_col="uplift",
            propensity_col="propensity",
            treatment_val=1,
            control_val=0,
            model_key=model_path,
            save_path=aucc_save_path,
            eps=0.02,
            n_bins=100,
            use_estimated_propensity_if_missing=True,
            feature_prefix="x__",
            x_axis="policy_cost",
            save_debug_csv_path=f"{model_path}/result/aucc_nonrct_ips_snips_debug.csv",
        )

        print(f"模型 {model_path} 的 Non-RCT IPS/SNIPS-AUCC 分数为: {aucc_score_ips:.6f}")


# ==================== 统一绘图 ====================
json_file_path = aucc_save_path
output_image_path = f"result/aucc_curves_nonrct_ips_snips_{current_time}.pdf"

plot_aucc_from_json(
    json_file_path,
    output_image_path,
    model_names=model_paths_DFCL + model_paths_else,
    model_name_map=model_name_map,
)
