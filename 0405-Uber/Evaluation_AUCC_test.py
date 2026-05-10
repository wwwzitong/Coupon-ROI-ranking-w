#!/usr/bin/env python
# coding: utf-8

# # Evaluation
# benchmark 包含DRM+DFL+DFCL*3

# In[1]:


from __future__ import print_function, absolute_import, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有 GPU，自然不会加载 CUDA。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息（隐藏 INFO 和 WARNING）

import sys
import io
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# ==================== 设置随机种子确保可复现性 ====================
def set_seeds(seed=42):
    """
    设置所有随机种子以确保实验可复现
    Args:
        seed: 随机种子值，默认为42
    """
    # 设置Python随机种子
    random.seed(seed)
    
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置TensorFlow随机种子
    tf.random.set_seed(seed)
    # 设置操作确定性（可能影响性能但提高可复现性）
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # 设置PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"已设置随机种子: {seed}")

set_seeds(42)  # 你可以更改为任何固定值

# from fsfc_mine_mx2 import * #自行生成fsfc文件（脚本放在data_flow中）
# from data_utils_mx2 import *

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from data_utils_UScensus import *

# 将输出保存到文件

import threading
from contextlib import contextmanager

# ===== Non-RCT AUCC correction imports =====
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except ImportError:
    LogisticRegression = None
    StandardScaler = None
    make_pipeline = None


class TeeStream(io.TextIOBase):
    """把写入内容同时写到多个 text stream（例如：原终端 + 文件）"""
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
        # 保留 isatty 行为，避免某些库判断错
        return any(getattr(st, "isatty", lambda: False)() for st in self.streams)


@contextmanager
def tee_output(filepath, mode="a", encoding="utf-8"):
    """
    在 with 块内，将 stdout/stderr 同时输出到终端和文件。
    mode: "a" 追加, "w" 覆盖
    """
    old_out, old_err = sys.stdout, sys.stderr
    # buffering=1: 行缓冲，尽量边跑边落盘
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


config = {
    'eval_data': '../data/census1990_test.csv',
    'batch_size': 1024,
    'max_batches_for_eval':79,
    'aucc_save_path': "result/result_aucc.json", #保存好坐标点，以便后续画图
    'auuc_save_path': "result/result_auuc.json" #保存好坐标点，以便后续画图
}
# # 训练集上试试
# config = {
#     'eval_data': 'data/train_set/part-*',
#     'batch_size': 1024*16,
#     'max_batches_for_eval':50,
#     'aucc_save_path': "result/result_aucc.json", #保存好坐标点，以便后续画图
#     'auuc_path': "result/auuc.json" #保存好坐标点，以便后续画图
# }


# In[4]:


# --- 2. 加载测试集 ---
dataset = CSVData()
eval_samples = dataset.prepare_dataset(
    config['eval_data'], 
    phase='eval', 
    batch_size=config['batch_size'], # 一次性加载所有数据进行评估 =None
    shuffle=False
)

# --- Step: 提取 drop_list 和 label_name_list ---
label_name_list = ['treatment','paid','cost']
drop_list = ['paid','cost']

# --- Step: 将 dataset 转换为 (features, labels) 格式 ---
def _to_features_labels(parsed_example):
    # 提取 features（从 feature_name_list 中）
    features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}
    labels = {}
    for name in label_name_list:
        value = parsed_example[name]
        labels[name] = value
    return features, labels  # 返回 (features, labels) 其中 labels 是 dict
# --- 应用 map 转换 ---
eval_samples = eval_samples.map(
    _to_features_labels,
    num_parallel_calls=4
).prefetch(1)

# In[6]:


# 步骤 3: 循环评估每个已保存的模型
model_paths_DFCL = [
    # "./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3_seed44",
    # "./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10_seed44",
    # "./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5_seed44",
    # "./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100_seed44",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed44",
    
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=0.1_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=0.5_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=2_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=3_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=4_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=5_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=10_tau=1.0_seed42",

    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=0.1_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=0.5_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=1_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=2_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=3_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=4_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=5_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=10_seed42",


    # "./model/SLearner_mse_mean_bs256_step500_lr1e-3_clip=5e3",
    # "./model/EcomDFCL_v3_mse_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10",
    # "./model/EcomDFCL_v3_mse_3erl_bs256_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5",
    # "./model/EcomDFCL_v3_mse_4ifdl_bs256_step500_lr1e-3_clip=100_alpha=100",
    # "./model/EcomDFCL_regretNet_rplusc_mse_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",

    # "./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3",
    # "./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10",
    # "./model/EcomDFCL_v3_wce_3erl_bs256_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5",
    # "./model/EcomDFCL_v3_wce_4ifdl_bs256_step500_lr1e-3_clip=100_alpha=100",

    "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",

]
model_paths_else = [



]

model_name_map = {
    # "./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3": "MTP",
    # "./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10": "DFCL-PL",
    # "./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5": "DFCL-MER",
    # "./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100": "DFCL-IFD",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0": "CC-DFL(Ours)",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0": "CC-DFL",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed39":"seed39",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40":"seed40",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed41":"seed41",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0":"seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed43":"seed43",

    # "./model/rplusc_wce_bs256_lr1e-4_clip=5e3_max=1_tau=1_rho=0": "CC-DFL w/o QP",
    # "./model/rplusc_wce_bs256_lr1e-4_clip=5e3_max=0_tau=1_rho=0": "CC-DFL w/o CP",
}

# In[7]:


treatment_order = [1, 0] #处理组为15off，另一组是空白组
ratios = [i / 100.0 for i in range(5, 105, 5)]
aucc_save_path = config['aucc_save_path']
auuc_save_path = config['auuc_save_path']


# 生成当前时间字符串
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ## AUCC

## 0508新增开始 ##
def _safe_np_1d(x, dtype=float):
    """把 tensor / list / Series 稳定转成一维 numpy array。"""
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(-1)


def _clip_propensity(p, eps=0.02):
    """
    对 propensity 做截断，避免 IPS/DR 权重爆炸。
    eps 可以根据 overlap 情况调，比如 0.01 / 0.02 / 0.05。
    """
    p = _safe_np_1d(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def estimate_propensity_from_feature_columns(
    df,
    treatment_col='treatment',
    feature_prefix='x__',
    propensity_col='propensity',
    eps=0.02,
):
    """
    使用 eval_df 中以 feature_prefix 开头的特征列估计 e(x)=P(T=1|X)。

    注意：
    1. 如果你的数据里有真实 logged propensity，优先直接用真实 propensity，不要估计。
    2. 如果没有 logged propensity，才用这个函数估计。
    3. 该函数要求 eval_df 中已经带有 x__ 开头的特征列。
    """
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    if len(feature_cols) == 0:
        raise ValueError(
            f"没有找到以 {feature_prefix!r} 开头的特征列，无法估计 propensity。"
            "请先按“修改位置 3”把 features_batch 收集进 eval_df，"
            "或者如果数据有真实 logged propensity，请直接在 eval_df 中提供 propensity 列。"
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
            solver='lbfgs',
            class_weight='balanced',
            n_jobs=None,
        )
    )
    clf.fit(X, y)

    p = clf.predict_proba(X)[:, 1]
    df[propensity_col] = _clip_propensity(p, eps=eps)

    print(
        f"[Propensity] 使用 {len(feature_cols)} 个特征估计完成："
        f"min={df[propensity_col].min():.4f}, "
        f"p01={df[propensity_col].quantile(0.01):.4f}, "
        f"mean={df[propensity_col].mean():.4f}, "
        f"p99={df[propensity_col].quantile(0.99):.4f}, "
        f"max={df[propensity_col].max():.4f}"
    )
    return df


def add_dr_pseudo_outcomes_for_nonrct(
    df,
    reward_col='paid',
    cost_col='cost',
    treatment_col='treatment',
    propensity_col='propensity',
    mu1_reward_col='treat_paid',
    mu0_reward_col='ctrl_paid',
    mu1_cost_col='treat_cost',
    mu0_cost_col='ctrl_cost',
    eps=0.02,
    output_reward_col='dr_delta_reward',
    output_cost_col='dr_delta_cost',
):
    """
    为非 RCT 评估构造 DR pseudo outcome。

    对收益：
        DR_R = μ1_R(x) - μ0_R(x)
               + T/e(x) * (R - μ1_R(x))
               - (1-T)/(1-e(x)) * (R - μ0_R(x))

    对成本：
        DR_C = μ1_C(x) - μ0_C(x)
               + T/e(x) * (C - μ1_C(x))
               - (1-T)/(1-e(x)) * (C - μ0_C(x))

    其中 μ1/μ0 这里直接用你模型输出里的 treat_paid/ctrl_paid/treat_cost/ctrl_cost。
    """
    required_cols = [
        reward_col, cost_col, treatment_col, propensity_col,
        mu1_reward_col, mu0_reward_col, mu1_cost_col, mu0_cost_col
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DR 校正缺少必要列: {missing}")

    out = df.copy()

    t = out[treatment_col].astype(int).values
    e = _clip_propensity(out[propensity_col].values, eps=eps)

    r = out[reward_col].astype(float).values
    c = out[cost_col].astype(float).values

    mu1_r = out[mu1_reward_col].astype(float).values
    mu0_r = out[mu0_reward_col].astype(float).values
    mu1_c = out[mu1_cost_col].astype(float).values
    mu0_c = out[mu0_cost_col].astype(float).values

    out[output_reward_col] = (
        (mu1_r - mu0_r)
        + t / e * (r - mu1_r)
        - (1 - t) / (1 - e) * (r - mu0_r)
    )

    out[output_cost_col] = (
        (mu1_c - mu0_c)
        + t / e * (c - mu1_c)
        - (1 - t) / (1 - e) * (c - mu0_c)
    )

    print(
        f"[DR] pseudo outcome 完成："
        f"{output_reward_col} mean={out[output_reward_col].mean():.6f}, "
        f"{output_cost_col} mean={out[output_cost_col].mean():.6f}"
    )
    return out

def calculate_and_save_aucc_nonrct_dr(
    df,
    reward_col='paid',
    cost_col='cost',
    treatment_col='treatment',
    uplift_col='uplift',
    propensity_col='propensity',
    treatment_val=1,
    control_val=0,
    model_key=None,
    save_path=None,
    eps=0.02,
    n_bins=100,
    use_estimated_propensity_if_missing=True,
    feature_prefix='x__',
    x_axis='fraction',   # 推荐先用 'fraction'；如需 cost 曲线再改成 'cost'
    save_debug_csv_path=None,
):
    """
    Non-RCT corrected AUCC / AUUC-style curve.

    x_axis:
        'fraction': 横轴为被选中人群比例 q，最稳定，推荐用于检查排序质量。
        'cost':     横轴为累计 DR incremental cost / final cost，不做强制 clip。
                    该曲线可能超出 [0,1]，这是 cost frontier 的真实表现。

    注意：
    - 本函数只对 ΔR / ΔC 做 DR 校正。
    - 不强行把 y clip 到 [0,1]，否则会出现 y=1 水平线。
    - 不强行让 x 单调，否则会掩盖成本估计中的真实回折。
    """
    if save_path is None:
        save_path = aucc_save_path
    if model_key is None:
        model_key = model_path

    df_filtered = df[
        (df[treatment_col] == control_val) | (df[treatment_col] == treatment_val)
    ].copy()

    df_filtered[treatment_col] = (df_filtered[treatment_col] == treatment_val).astype(int)

    # 1. 准备 propensity
    if propensity_col not in df_filtered.columns:
        if not use_estimated_propensity_if_missing:
            raise ValueError(
                f"非 RCT 评估需要 {propensity_col!r} 列。"
                "如果没有真实 logged propensity，请先提供该列，"
                "或者设置 use_estimated_propensity_if_missing=True。"
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

    # 2. 构造 DR pseudo outcome
    df_dr = add_dr_pseudo_outcomes_for_nonrct(
        df_filtered,
        reward_col=reward_col,
        cost_col=cost_col,
        treatment_col=treatment_col,
        propensity_col=propensity_col,
        eps=eps,
        output_reward_col='dr_delta_reward',
        output_cost_col='dr_delta_cost',
    )

    # 3. 按模型分数排序
    df_sorted = df_dr.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        raise ValueError("AUCC 输入为空。")

    # 4. 逐点累计 DR 增量
    cum_gain = df_sorted['dr_delta_reward'].astype(float).cumsum().values
    cum_cost = df_sorted['dr_delta_cost'].astype(float).cumsum().values
    frac = np.arange(1, n + 1, dtype=float) / n

    # 5. 分桶降噪：按排序位置取点，不按 cost 排序
    if n_bins is not None and n_bins > 0 and n > n_bins:
        idx = np.unique(np.ceil(np.linspace(1, n, n_bins)).astype(int) - 1)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        gain_plot = cum_gain[idx]
        cost_plot = cum_cost[idx]
        frac_plot = frac[idx]
    else:
        gain_plot = cum_gain
        cost_plot = cum_cost
        frac_plot = frac

    # 6. 加原点
    gain_plot = np.r_[0.0, gain_plot]
    cost_plot = np.r_[0.0, cost_plot]
    frac_plot = np.r_[0.0, frac_plot]

    final_gain = gain_plot[-1]
    final_cost = cost_plot[-1]

    if np.isclose(final_gain, 0.0):
        print("[AUCC-DR] final_gain 接近 0，无法做 gain 归一化。")
        y = gain_plot
    else:
        y = gain_plot / final_gain

    if x_axis == 'fraction':
        x = frac_plot
        x_label = 'Selected Fraction'
    elif x_axis == 'cost':
        if np.isclose(final_cost, 0.0):
            print("[AUCC-DR] final_cost 接近 0，无法做 cost 归一化，使用原始累计 cost。")
            x = cost_plot
        else:
            x = cost_plot / final_cost
        x_label = 'Normalized DR Incremental Cost'
    else:
        raise ValueError("x_axis 只能是 'fraction' 或 'cost'。")

    # 7. 不要 clip，不要 maximum.accumulate
    # 只清理 nan/inf
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # fraction 横轴天然单调；cost 横轴可能不单调，trapz 只适合单调 x
    if x_axis == 'fraction':
        aucc_score = float(np.trapz(y, x))
    else:
        # cost 曲线如果 x 非单调，不建议直接解释 trapz 面积
        if np.any(np.diff(x) < 0):
            print(
                "[AUCC-DR] 警告：cost 横轴非单调，np.trapz 面积可能没有标准 AUCC 含义。"
                "建议先用 x_axis='fraction' 做标准排序评估。"
            )
        aucc_score = float(np.trapz(y, x))

    # 8. 保存 debug
    if save_debug_csv_path:
        debug_df = df_sorted.copy()
        debug_df['cum_dr_gain'] = cum_gain
        debug_df['cum_dr_cost'] = cum_cost
        debug_df['selected_fraction'] = frac
        debug_df.to_csv(save_debug_csv_path, index=False, encoding='utf-8-sig')
        print(f"[AUCC-DR] debug 明细已保存: {save_debug_csv_path}")

    # 9. 保存 JSON
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    all_results[model_key] = {
        'aucc_score': aucc_score,
        'x_coords': x.tolist(),
        'y_coords': y.tolist(),
        'method': 'nonrct_dr_no_clip',
        'x_axis': x_axis,
        'x_label': x_label,
        'propensity_col': propensity_col,
        'eps': eps,
        'final_dr_gain': float(final_gain),
        'final_dr_cost': float(final_cost),
        'y_min': float(np.min(y)) if len(y) else None,
        'y_max': float(np.max(y)) if len(y) else None,
        'x_min': float(np.min(x)) if len(x) else None,
        'x_max': float(np.max(x)) if len(x) else None,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    print(
        f"[AUCC-DR] 保存成功: {save_path}, "
        f"score={aucc_score:.6f}, "
        f"x_axis={x_axis}, "
        f"x_range=({np.min(x):.4f}, {np.max(x):.4f}), "
        f"y_range=({np.min(y):.4f}, {np.max(y):.4f}), "
        f"final_dr_gain={final_gain:.6f}, "
        f"final_dr_cost={final_cost:.6f}"
    )

    return aucc_score

## 0508新增结束 ##


def strict_aucc_algorithm2(df, reward_col='paid', cost_col='cost', treatment_col='treatment', uplift_col='uplift', bins=100):
    """
    reference：https://bytedance.larkoffice.com/docx/URpyd2iS9o8puxxvD7JcS3jUnkb?bk_entity_id=enterprise_7332387637669888004
    逐点排序并作图，无其他过滤逻辑。很考验数据本身的质量和模型的水平
    """
    # Step 1: 按置信分数（即uplift_col）降序排列
    df = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = df.shape[0]

    # Step 2: 初始化 S, ΔC_prev
    S = 0
    delta_C_prev = 0

    # 提前准备掩码
    treat_mask = df[treatment_col] == 1
    ctrl_mask = df[treatment_col] == 0

    # 初始化累计和
    treat_reward_cumsum = (df[reward_col] * treat_mask).cumsum()
    ctrl_reward_cumsum = (df[reward_col] * ctrl_mask).cumsum()
    treat_cost_cumsum = (df[cost_col] * treat_mask).cumsum()
    ctrl_cost_cumsum = (df[cost_col] * ctrl_mask).cumsum()

    # 预计算 ΔR_k 和 ΔC_k 序列
    delta_R_list = treat_reward_cumsum - ctrl_reward_cumsum  # ΔR_k
    delta_C_list = treat_cost_cumsum - ctrl_cost_cumsum      # ΔC_k

    # Step 3-10: 主循环积分
    # S = ∑_{k=1}^{n} ΔR_k × (ΔC_k - ΔC_{k-1})
    for k in range(n):
        delta_R_k = delta_R_list.iloc[k]
        delta_C_k = delta_C_list.iloc[k]
        S += delta_R_k * (delta_C_k - delta_C_prev)
        delta_C_prev = delta_C_k

    # Step 8: 归一化分母 S_normal（用最大 ΔR 点 × 最大 ΔC 点）
    delta_R_max = delta_R_list.iloc[-1]
    delta_C_max = delta_C_list.iloc[-1]
    S_normal = delta_R_max * delta_C_max

#     # Step 11: 返回标准化 AUCC
#     return S / S_normal if S_normal != 0 else np.nan
    aucc_score = S / S_normal if S_normal != 0 else np.nan

    # --- 新增：生成绘图坐标 ---
    # 归一化坐标
    # 检查分母是否为0，避免除零错误
    norm_x_coords = [0] + (delta_C_list / delta_C_max).tolist() if delta_C_max != 0 else [0] * (n + 1)
    norm_y_coords = [0] + (delta_R_list / delta_R_max).tolist() if delta_R_max != 0 else [0] * (n + 1)
    
    
     # 1. 读取现有数据
    try:
        with open(aucc_save_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    # 2. 更新数据
    all_results[model_path] = {
        'aucc_score': aucc_score,
        'x_coords': norm_x_coords,
        'y_coords': norm_y_coords
    }

    # 3. 写回文件
    with open(aucc_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print("\n保存成功！后续可加载此文件直接绘制 AUCC 曲线，无需重新计算。")
    return aucc_score


# ## AUCC公司另外一个版本

# In[10]:


def calculate_and_save_aucc(df, reward_col='paid', cost_col='cost', treatment_col='treatment', uplift_col='uplift', uplift_gmv_col='uplift_gmv', uplift_cost_col='uplift_cost', treatment_val=1, control_val=0, n_bins=100, output_dip_samples_path="result/dip_samples.csv"):
    '''
    公司另外一个版本的AUCC,分bins去作图，以bins中的第一个user作为落点依据。图会更加的平滑。无其他过滤逻辑。
    '''
    # --- 第1步: 筛选实验组和控制组数据 ---
    df_filtered = df[(df[treatment_col] == control_val) | (df[treatment_col] == treatment_val)].copy()

    # --- 第2步: 计算排序指标 (pred_roi) 并排序 PS：这里把公司的给改了，当前的排序指标只对应ratio=1的情况，我还是倾向于累加模拟积分 ---
    df_filtered['pred_roi'] = df_filtered[uplift_col]
    
    df_sorted = df_filtered.sort_values('pred_roi', ascending=False).reset_index(drop=True)
    # 将索引从1开始，方便后续计算累积用户数
    df_sorted.index = df_sorted.index + 1

    # --- 第3步: 计算累积uplift (delta_gain, delta_cost) ---
    is_treatment = (df_sorted[treatment_col] == treatment_val)
    
    cumsum_tr = is_treatment.cumsum()
    cumsum_ct = df_sorted.index.values - cumsum_tr
    
    # 为避免除以0，将累积数为0的替换为NaN，后续计算平均值时会自动忽略
    cumsum_tr_safe = cumsum_tr.replace(0, np.nan)
    cumsum_ct_safe = cumsum_ct.replace(0, np.nan)

    cumsum_gain_tr = (df_sorted[reward_col] * is_treatment).cumsum()
    cumsum_gain_ct = (df_sorted[reward_col] * ~is_treatment).cumsum()
    cumsum_cost_tr = (df_sorted[cost_col] * is_treatment).cumsum()
    cumsum_cost_ct = (df_sorted[cost_col] * ~is_treatment).cumsum()

    # 计算累积uplift
    df_sorted['delta_gain'] = (cumsum_gain_tr / cumsum_tr_safe - cumsum_gain_ct / cumsum_ct_safe).fillna(0) * df_sorted.index.values
    df_sorted['delta_cost'] = (cumsum_cost_tr / cumsum_tr_safe - cumsum_cost_ct / cumsum_ct_safe).fillna(0) * df_sorted.index.values

    # --- 第4步: 按 delta_cost 进行分桶 ---
    # 完全遵循 metric.py 中的分桶逻辑
    df_sorted['cost_bin'] = pd.cut(df_sorted['delta_cost'], bins=n_bins, labels=False, include_lowest=True)
    
    # 取每个桶的第一个点作为曲线的关键点
    df_binned = df_sorted.groupby('cost_bin').first()
    
    # 确保曲线的终点是全体用户的最终uplift值
    last_row = df_sorted.iloc[[-1]]
    df_binned = pd.concat([df_binned, last_row]).reset_index() # 使用reset_index保留原始的index
    df_binned = df_binned.drop_duplicates(subset=['delta_cost'], keep='first').sort_values('delta_cost')

    # --- 新增：检测并保存 AUCC 曲线下降区间的样本 ---
    if output_dip_samples_path and len(df_binned) > 1:
        all_dip_samples = []
        df_binned_reset = df_binned.reset_index(drop=True)
        for i in range(len(df_binned_reset) - 1):
            current_gain = df_binned_reset.loc[i, 'delta_gain']
            next_gain = df_binned_reset.loc[i+1, 'delta_gain']

            if next_gain < current_gain:
                # 发现一个下降区间
                start_original_idx = df_binned_reset.loc[i, 'index']
                end_original_idx = df_binned_reset.loc[i+1, 'index']
                
                print(f"检测到 AUCC 曲线下降区间: 从第 {i} 个点到第 {i+1} 个点。")
                print(f"  - Delta Gain: {current_gain:.2f} -> {next_gain:.2f}")
                print(f"  - 对应原始排序后样本范围 (索引从1开始): {start_original_idx} -> {end_original_idx}")

                # 提取这部分样本 (df_sorted的索引是从1开始的)
                dip_samples = df_sorted.loc[start_original_idx:end_original_idx].copy()
                dip_samples['dip_segment'] = f'dip_{i}_to_{i+1}'
                all_dip_samples.append(dip_samples)
        
        if all_dip_samples:
            combined_dip_samples = pd.concat(all_dip_samples)
            combined_dip_samples.to_csv(output_dip_samples_path, index=False, encoding='utf-8-sig')
            print(f"\n已将所有下降区间的样本数据保存至: {output_dip_samples_path}")

    # --- 第5步: 计算AUCC分数 ---
    final_delta_gain = df_binned['delta_gain'].iloc[-1]
    
    if final_delta_gain == 0 or len(df_binned) <= 1:
        aucc_score = 0.5  # 如果没有增益或数据不足，模型效果等同于随机
    else:
        # 使用 metric.py 中的归一化面积公式
        # (曲线下面积) / (最大增益) / (桶数)
        aucc_score = (df_binned['delta_gain'].sum() - final_delta_gain / 2) / final_delta_gain / n_bins

    # --- 第6步: 准备并保存绘图数据到JSON文件 ---
    final_delta_cost = df_binned['delta_cost'].iloc[-1]

    # 归一化坐标轴到[0, 1]区间，用于绘图
    norm_x_coords = [0] +(df_binned['delta_cost'] / final_delta_cost).tolist() if final_delta_cost > 0 else [0.0] * len(df_binned)
    norm_y_coords = [0] +(df_binned['delta_gain'] / final_delta_gain).tolist() if final_delta_gain > 0 else [0.0] * len(df_binned)

     # 1. 读取现有数据 看清楚路径
    try:
        with open("result/result_aucc_v2.json", 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    # 2. 更新数据
    all_results[model_path] = {
        'aucc_score': aucc_score,
        'x_coords': norm_x_coords,
        'y_coords': norm_y_coords
    }
    
    with open("result/result_aucc_v2.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    return aucc_score


# In[11]:


def get_aucc_plot(pdf, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='uplift', treatment_index=1, model_path='unknown_model'):
    """
    计算AUCC
    :param pdf: 计算aucc的pandas df
    :param treatment_col: treatment列名
    :param gain_col: 收益列名
    :param cost_col: 成本列名
    :param pred_roi_col: roi排序列名
    :param treatment_index: 对哪个treatment计算
    :return: aucc的值
    """
    aucc_dict = {}
    aucc_dict[pred_roi_col] = {}
    df = pdf[(pdf[treatment_col] == 0) | (pdf[treatment_col] == treatment_index)].reset_index(drop=True)[[gain_col, cost_col, pred_roi_col, treatment_col]]
    df = df.sort_values(pred_roi_col, ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    cumsum_tr = (df[treatment_col] != 0).cumsum().replace(0, np.nan)
    # print(cumsum_tr)
    cumsum_ct = (df.index.values - cumsum_tr).replace(0, np.nan)
    cumsum_gain_tr = (df[gain_col] * (df[treatment_col] != 0)).cumsum()
    cumsum_gain_ct = (df[gain_col] * (df[treatment_col] == 0)).cumsum()
    cumsum_cost_tr = (df[cost_col] * (df[treatment_col] != 0)).cumsum()
    cumsum_cost_ct = (df[cost_col] * (df[treatment_col] == 0)).cumsum()
    df["delta_gain"] = (cumsum_gain_tr / cumsum_tr - cumsum_gain_ct / cumsum_ct).fillna(0) * df.index.values
    df["delta_cost"] = (cumsum_cost_tr / cumsum_tr - cumsum_cost_ct / cumsum_ct).fillna(0) * df.index.values 
    
    # --- 增加的归一化逻辑 ---
    # 获取总的增量收益和成本，用于归一化
    total_delta_gain = df["delta_gain"].iloc[-1] if not df.empty else 0
    total_delta_cost = df["delta_cost"].iloc[-1] if not df.empty else 0

    # 归一化 delta_gain 和 delta_cost 到 [0, 1] 区间，并处理总增量为0的情况
    df['norm_delta_gain'] = df['delta_gain'] / total_delta_gain if total_delta_gain != 0 else 0
    df['norm_delta_cost'] = df['delta_cost'] / total_delta_cost if total_delta_cost != 0 else 0
    # 使用归一化后的值进行绘图
    plt.plot(df['norm_delta_cost'], df['norm_delta_gain'], label='model_pred')
    # 在归一化空间中，随机曲线是一条从(0,0)到(1,1)的对角线
    plt.plot(df['norm_delta_cost'], df['norm_delta_cost'], label='random', linestyle='--')
    plt.legend()
    plt.xlabel("Normalized Cumulative Cost")
    plt.ylabel("Normalized Cumulative Gain")
    plt.title(f"Normalized AUCC Curve ({pred_roi_col})")

    # plt.show()
    # --- 修改开始：保存图像逻辑 ---
    current_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_save_path = f"{model_path}/result/aucc_{pred_roi_col}_treat{treatment_index}_{current_ts}.png"

    plt.savefig(plot_save_path)
    plt.close() # 关闭图像，释放内存
    print(f"AUCC Plot ({pred_roi_col}) 已保存至: {plot_save_path}")
    # --- 修改结束 ---

    aucc = np.trapz(df['norm_delta_gain'], df['norm_delta_cost'])
    random_area = np.trapz(df['norm_delta_cost'], df['norm_delta_cost'])
    aucc_score = aucc/2/random_area
    aucc_dict[pred_roi_col]['score'] = aucc_score
    aucc_dict[pred_roi_col]['treatment'] = treatment_index
    aucc_dict[pred_roi_col]['random_score'] = 0.5
    print(aucc_dict)
    return aucc_dict




# --- 评估流程 (已完善) 倾向于每一个model单独计算完metric，进行数据保留后统一画图（因为模型输出接口不一致）---
print("开始评估流程...")

for model_path in model_paths_DFCL: 
    print(f"\n{'='*20} 正在评估模型: {model_path} {'='*20}")

    result_dir = Path(f"./{model_path}/result")   # 或 r".\result"
    result_dir.mkdir(parents=True, exist_ok=True)  # 不存在就创建，存在也不报错
    print("result文件夹已创建...")
    
    # 2.1 加载模型并进行预测
    print("加载模型并进行预测...")
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # 假设模型是 Keras SavedModel 格式
        model = tf.keras.models.load_model(model_path, compile=False) 
        
    # 初始化用于存储所有批次结果的列表
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
    all_feature_batches = [] # Non-RCT: 收集特征用于估计 propensity
    
    print("开始分批次进行预测...")
    max_batches_for_eval = config["max_batches_for_eval"]  # 示例：限制为100个批次

    # 遍历评估数据集的每个批次
    for i, (features_batch, labels_batch) in enumerate(eval_samples):
        # 检查是否达到批次数限制
        if max_batches_for_eval is not None and i >= max_batches_for_eval:
            print(f"已达到最大评估批次数 {max_batches_for_eval}，停止预测。")
            break
        # 在当前批次上进行预测
        predictions_logits= model.predict(features_batch)

        # ！！提取标签和计算Uplift
        pred_dict = {key: tf.exp(tf.minimum(logit, 10.0)) for key, logit in predictions_logits.items()}
        pred_paid_treat = pred_dict['paid_treatment_1']
        pred_cost_treat = pred_dict['cost_treatment_1']
        
        pred_paid_ctrl = pred_dict['paid_treatment_0']
        pred_cost_ctrl = pred_dict['cost_treatment_0']
        
        # 计算uplift
        num_samples = len(pred_paid_treat)
        total_uplift_per_sample = np.zeros(num_samples)

        for r in ratios:
            uplift = pred_paid_treat - r*pred_cost_treat
            total_uplift_per_sample += uplift
        integrated_uplift_per_sample = total_uplift_per_sample / len(ratios)
        
        pred_paid_uplift = pred_dict['paid_treatment_1'] - pred_dict['paid_treatment_0']
        pred_cost_uplift = pred_dict['cost_treatment_1'] - pred_dict['cost_treatment_0']
        roi_tensor = tf.where(
            pred_cost_uplift > 0,                      # 条件
            tf.math.divide_no_nan(pred_paid_uplift, pred_cost_uplift), # 条件为 True 时的值
            tf.zeros_like(pred_paid_uplift)            # 条件为 False 时的值
        )
        roi = roi_tensor.numpy()

        # 收集当前批次的预测uplift和真实标签
        all_uplifts.append(integrated_uplift_per_sample)
        # 新增uplift逻辑
        all_uplift_paid.append(pred_paid_uplift)
        all_uplift_cost.append(pred_cost_uplift)
        # 新增结束
        all_rois.append(roi)
        all_treat_paid.append(pred_paid_treat)
        all_treat_cost.append(pred_cost_treat)
        all_ctrl_paid.append(pred_paid_ctrl)
        all_ctrl_cost.append(pred_cost_ctrl)
        all_paid_labels.append(labels_batch['paid'].numpy())
        all_cost_labels.append(labels_batch['cost'].numpy())
        # all_treatment_labels.append(labels_batch['_treatment_index'].numpy())
        all_treatment_labels.append(labels_batch['treatment'].numpy())
        # ===== Non-RCT: 收集当前 batch 的数值特征 =====
        feature_batch_np = {}
        for fname, fval in features_batch.items():
            arr = fval.numpy() if hasattr(fval, "numpy") else np.asarray(fval)

            # 只收集一维数值特征；如果有 embedding/list 特征，先跳过，避免 DataFrame 维度错误。
            arr = np.asarray(arr)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                feature_batch_np[f"x__{fname}"] = arr

        all_feature_batches.append(pd.DataFrame(feature_batch_np))

    print("所有批次预测完成，正在整合结果...")
    # 将所有批次的结果（list of arrays）拼接成一个大的Numpy数组
    final_uplifts = np.concatenate(all_uplifts)
    final_uplift_paid = np.concatenate(all_uplift_paid) # new
    final_uplift_cost = np.concatenate(all_uplift_cost) # new
    final_rois = np.concatenate(all_rois)
    final_treat_paid = np.concatenate(all_treat_paid)
    final_treat_cost = np.concatenate(all_treat_cost)
    final_ctrl_paid = np.concatenate(all_ctrl_paid)
    final_ctrl_cost = np.concatenate(all_ctrl_cost)
    final_paid = np.concatenate(all_paid_labels)
    final_cost = np.concatenate(all_cost_labels)
    final_treatment = np.concatenate(all_treatment_labels)
    
    # 6. 整合为DataFrame
    print("正在将所有结果整合到DataFrame...")
    eval_df = pd.DataFrame({
        'paid': final_paid,
        'cost': final_cost,
        'treatment': final_treatment,
        'uplift': final_uplifts,
        'roi':final_rois,
        'treat_paid': final_treat_paid,
        'treat_cost': final_treat_cost,
        'ctrl_paid': final_ctrl_paid,
        'ctrl_cost': final_ctrl_cost,
        'uplift_paid': final_uplift_paid, # new
        'uplift_cost': final_uplift_cost, # new
    })

    # ===== Non-RCT: 把用于估计 propensity 的特征拼到 eval_df =====
    if len(all_feature_batches) > 0:
        feature_df = pd.concat(all_feature_batches, axis=0, ignore_index=True)
        feature_df = feature_df.reset_index(drop=True)

        if len(feature_df) != len(eval_df):
            raise ValueError(
                f"feature_df 行数 {len(feature_df)} 与 eval_df 行数 {len(eval_df)} 不一致，"
                "请检查 batch 收集逻辑。"
            )

        eval_df = pd.concat([eval_df.reset_index(drop=True), feature_df], axis=1)
        print(f"[Non-RCT] 已拼接 propensity 特征列数量: {feature_df.shape[1]}")

    
    with tee_output(f"{model_path}/eval.log", mode="a", encoding="utf-8"):
        # 打印结果DataFrame的前几行以供查验
        print("\n评估结果DataFrame示例:")
        print(eval_df.head())
        eval_df['treatment'] = eval_df['treatment'].astype(int)
        
        # # 7. 计算 AUCC 并获取绘图数据
        # print("正在计算 AUCC 指标...")
        # aucc_score = strict_aucc_algorithm2(df=eval_df)
        # print(f"模型 {model_path} 的 AUCC 分数为: {aucc_score:.6f}")
        # aucc_score_2 = calculate_and_save_aucc(df=eval_df)
        # print(f"模型 {model_path} 的 AUCC公司版本 分数为: {aucc_score_2:.6f}")


        # 7. 计算 Non-RCT corrected AUCC，并保存绘图数据
        print("正在计算 Non-RCT DR-corrected AUCC 指标...")

        # 如果你的 eval_df 已经有真实 logged propensity，比如列名叫 logged_propensity，
        # 就先执行：
        # eval_df['propensity'] = eval_df['logged_propensity']
        #
        # 如果没有真实 propensity，下面函数会用 x__ 特征列估计 propensity。
        aucc_score_dr = calculate_and_save_aucc_nonrct_dr(
            df=eval_df,
            reward_col='paid',
            cost_col='cost',
            treatment_col='treatment',
            uplift_col='uplift',
            propensity_col='propensity',
            treatment_val=1,
            control_val=0,
            model_key=model_path,
            save_path=aucc_save_path,
            eps=0.02,
            n_bins=100,
            use_estimated_propensity_if_missing=True,
            feature_prefix='x__',
            x_axis='cost',   # 先用这个
            save_debug_csv_path=f"{model_path}/result/aucc_nonrct_dr_debug.csv",
        )

        print(f"模型 {model_path} 的 Non-RCT DR-AUCC 分数为: {aucc_score_dr:.6f}")

        # 可选：如果你仍想保留原始未校正 AUCC 作为对照，可以打开下面两行。
        # aucc_score_raw = strict_aucc_algorithm2(df=eval_df)
        # print(f"模型 {model_path} 的 Raw AUCC 分数为: {aucc_score_raw:.6f}")



        print("正在生成 AUCC Plot (Uplift)...")
        get_aucc_plot(eval_df, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='uplift', treatment_index=1, model_path=model_path)
        
        print("正在生成 AUCC Plot (ROI)...")
        get_aucc_plot(eval_df, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='roi', treatment_index=1, model_path=model_path)
        



from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

def plot_aucc_from_json(
    json_path: str,
    plot_path: str = 'aucc_comparison.pdf',
    model_names: Optional[List[str]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    fallback_to_basename: bool = True,
):
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from typing import Any, Dict, List, Optional

    legend_fs = 16
    axis_label_fs = legend_fs
    tick_fs = legend_fs

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
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

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({
        "font.size": legend_fs,
        "axes.labelsize": axis_label_fs,
        "xtick.labelsize": tick_fs,
        "ytick.labelsize": tick_fs,
        "legend.fontsize": legend_fs,
    })

    OTHER_COLORS = [
        "#1f77b4",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
    ]
    color_idx = 0

    base_lw = 2.5
    highlight_name = "CC-DFL"

    plt.figure(figsize=(7.2, 5.4))

    def build_feasible_curve(x_coords, y_coords):
        """
        只保留 x >= 0 的 feasible 区域；
        若线段穿过 x=0，则做线性插值补一个交点，避免断裂。
        """
        x_arr = np.asarray(x_coords, dtype=float)
        y_arr = np.asarray(y_coords, dtype=float)

        feasible_x = []
        feasible_y = []

        for i in range(len(x_arr) - 1):
            x1, y1 = x_arr[i], y_arr[i]
            x2, y2 = x_arr[i + 1], y_arr[i + 1]

            if x1 >= 0:
                if len(feasible_x) == 0 or (feasible_x[-1] != x1 or feasible_y[-1] != y1):
                    feasible_x.append(x1)
                    feasible_y.append(y1)

            # 穿过 x=0，补交点
            if (x1 < 0 and x2 > 0) or (x1 > 0 and x2 < 0):
                t = (0.0 - x1) / (x2 - x1)
                y0 = y1 + t * (y2 - y1)
                feasible_x.append(0.0)
                feasible_y.append(y0)

            if x2 >= 0:
                feasible_x.append(x2)
                feasible_y.append(y2)

        # 去重
        dedup_x, dedup_y = [], []
        for x, y in zip(feasible_x, feasible_y):
            if len(dedup_x) == 0 or (x != dedup_x[-1] or y != dedup_y[-1]):
                dedup_x.append(x)
                dedup_y.append(y)

        return dedup_x, dedup_y

    for model_key in ordered_keys:
        data = all_results.get(model_key, {})
        if not ('x_coords' in data and 'y_coords' in data and 'aucc_score' in data):
            print(f"模型 '{model_key}' 的数据不完整，跳过绘图。")
            continue

        display_name = model_name_map.get(model_key)
        if not display_name:
            display_name = Path(model_key).name if fallback_to_basename else model_key

        is_highlight = (display_name == highlight_name)

        if is_highlight:
            color = "red"
            raw_lw = base_lw * 1.2
            feasible_lw = raw_lw
            zorder_raw = 10
            zorder_feasible = 11
        else:
            color = OTHER_COLORS[color_idx % len(OTHER_COLORS)]
            color_idx += 1
            raw_lw = base_lw
            feasible_lw = base_lw
            zorder_raw = 1
            zorder_feasible = 2


        raw_x = np.asarray(data['x_coords'], dtype=float)
        raw_y = np.asarray(data['y_coords'], dtype=float)

        valid = np.isfinite(raw_x) & np.isfinite(raw_y)
        raw_x = raw_x[valid]
        raw_y = raw_y[valid]

        # 只有 fraction 横轴才排序；cost 横轴不要排序，否则会改变 policy path
        if data.get('x_axis', 'fraction') == 'fraction':
            order = np.argsort(raw_x)
            raw_x = raw_x[order]
            raw_y = raw_y[order]

        plt.plot(
            raw_x,
            raw_y,
            linestyle='-',
            linewidth=raw_lw,
            color=color,
            marker=None,
            zorder=zorder_raw,
            label=f'{display_name} DR-AUCC ({data["aucc_score"]:.4f})',
        )

    # Random
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=base_lw, label='Random', alpha=0.3)

    # 参考线：标出 feasible 分界
    plt.axvline(x=0.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)

    plt.xlabel('Incremental Cost(ΔC)', fontsize=axis_label_fs)
    plt.ylabel('Incremental Reward(ΔR)', fontsize=axis_label_fs)

    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)

    plt.legend(fontsize=legend_fs, frameon=True)
    plt.grid(True, alpha=0.3)

    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"AUCC 曲线对比图已保存至: {plot_path}")
    
#  'result_aucc.json'
json_file_path = aucc_save_path
output_image_path = f'result/aucc_curves_{current_time}.pdf'

# 调用函数生成图像
plot_aucc_from_json(json_file_path, output_image_path, model_names = model_paths_DFCL + model_paths_else, model_name_map=model_name_map)

#  'result_aucc_2.json'
json_file_path_2 = 'result/result_aucc_v2.json'
output_image_path_2 = f'result/aucc_curves_ByteDance_{current_time}.pdf'

# 调用函数生成图像
plot_aucc_from_json(json_file_path_2, output_image_path_2, model_names = model_paths_DFCL + model_paths_else, model_name_map=model_name_map)
