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
from data_utils_ECLIFT import *

# 将输出保存到文件

import threading
from contextlib import contextmanager

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
    'eval_data': '../data/ECLIFT_test.csv',
    'batch_size': 1024*16,
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
    # "./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3",
    # "./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10",
    # "./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5",
    # "./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",
    
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


    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=2.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=5.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=10_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=100_seed42",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=0.1_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=0.2_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=0.3_seed42",

    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=0.01_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=0.02_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=0.05_seed42",

    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=0.0001_seed42",
    # "./model/rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_rho=100_seed42",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_l10",
    "./model/EcomOneStepCCB_bs256_step500_lr1e-4_clip=5e3_tau=1.0",
    "./model/EcomOneStepCCB_bs256_step500_lr1e-3_clip=5e3_tau=1.0",
    "./model/EcomOneStepCCB_bs256_step500_lr1e-5_clip=5e3_tau=1.0",
    "./model/EcomOneStepCCB_bs256_step500_lr5e-5_clip=5e3_tau=1.0",
    "./model/EcomOneStepCCB_bs256_step500_lr5e-4_clip=5e3_tau=1.0",
    "./model/EcomOneStepCCB_bs256_step500_lr5e-3_clip=5e3_tau=1.0",




]
model_paths_else = [
    # "./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3",
    # "./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10",
    # "./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5",
    # "./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed39",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed41",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed43",
    
    # "./model/rplusc_wce_bs256_lr1e-4_clip=5e3_max=1_tau=1_rho=0",
    # "./model/rplusc_wce_bs256_lr1e-4_clip=5e3_max=0_tau=1_rho=0",

    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=0.5_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=0.7_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=0.9_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.2_seed42",
    # "./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.5_seed42",

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

# In[8]:



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

# In[9]:


def calculate_and_save_aucc_old(df, reward_col='paid', cost_col='cost', treatment_col='treatment', uplift_col='uplift', uplift_gmv_col='uplift_gmv', uplift_cost_col='uplift_cost', treatment_val=1, control_val=0, n_bins=100):
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


# ## AUUC

# In[12]:


def calculate_auuc(df, reward_col='cost', treatment_col='treatment', uplift_col='uplift'):
    """
    计算归一化的累计Uplift曲线下面积 (Normalized Cumulative AUUC)。
    """
    # Step 1: 按模型预测的 uplift 分数降序排列
    df_sorted = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n_total = len(df_sorted)

    # Step 2: 预先计算实验组和对照组的掩码及累计和，提高效率
    treat_mask = (df_sorted[treatment_col] == 1)
    ctrl_mask = (df_sorted[treatment_col] == 0)

    # 计算累计用户数
    n_treat_cumsum = treat_mask.cumsum()
    n_ctrl_cumsum = ctrl_mask.cumsum()

    # 计算累计收益
    reward_treat_cumsum = (df_sorted[reward_col] * treat_mask).cumsum()
    reward_ctrl_cumsum = (df_sorted[reward_col] * ctrl_mask).cumsum()

    # 防止除零错误
    n_treat_cumsum_safe = n_treat_cumsum.replace(0, 1e-9)
    n_ctrl_cumsum_safe = n_ctrl_cumsum.replace(0, 1e-9)

    # Step 3: 计算累计Uplift (Incremental Uplift)
    # 这是与“平均Uplift”方法的核心区别
    cumulative_uplift = (reward_treat_cumsum / n_treat_cumsum_safe - reward_ctrl_cumsum / n_ctrl_cumsum_safe) * (n_treat_cumsum + n_ctrl_cumsum)

    # Step 4: 归一化坐标轴
    population_fraction = np.arange(1, n_total + 1) / n_total
    x_coords = [0] + population_fraction.tolist()
    
    # Y轴通过除以总Uplift进行归一化
    total_uplift = cumulative_uplift.iloc[-1]
    if total_uplift != 0:
        y_coords_normalized = (cumulative_uplift / total_uplift).tolist()
    else:
        y_coords_normalized = [0] * n_total
    y_coords = [0] + y_coords_normalized
    
    # Step 5: 计算AUUC (曲线下面积) 和基线AUUC
    auuc_score = np.trapz(y=y_coords, x=x_coords)
    # 在归一化坐标系下，随机基线是y=x的对角线，其面积固定为0.5
    baseline_auuc = 0.5
    
    # 1. 读取现有数据
    try:
        with open(auuc_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    # 2. 更新数据
    all_results[model_path] = {
        'auuc_score': auuc_score,
        'baseline_auuc':baseline_auuc,
        'x_coords': x_coords,
        'y_coords': y_coords
    }

    # 3. 写回文件
    with open(auuc_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n模型 '{model_path}' 的 AUUC 结果已保存至 '{auuc_path}'")
    # --- 新增结束 ---


    return auuc_score, baseline_auuc


# In[13]:


def plot_auuc(df, reward_col='cost', treatment_col='treatment', uplift_col='uplift'):
    """
    计算归一化的累计Uplift曲线下面积 (Normalized Cumulative AUUC)。
    """
    # Step 1: 按模型预测的 uplift 分数降序排列
    df_sorted = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n_total = len(df_sorted)

    # Step 2: 预先计算实验组和对照组的掩码及累计和，提高效率
    treat_mask = (df_sorted[treatment_col] == 1)
    ctrl_mask = (df_sorted[treatment_col] == 0)

    # 计算累计用户数
    n_treat_cumsum = treat_mask.cumsum()
    n_ctrl_cumsum = ctrl_mask.cumsum()

    # 计算累计收益
    reward_treat_cumsum = (df_sorted[reward_col] * treat_mask).cumsum()
    reward_ctrl_cumsum = (df_sorted[reward_col] * ctrl_mask).cumsum()

    # 防止除零错误
    n_treat_cumsum_safe = n_treat_cumsum.replace(0, 1e-9)
    n_ctrl_cumsum_safe = n_ctrl_cumsum.replace(0, 1e-9)

    # Step 3: 计算累计Uplift (Incremental Uplift)
    # 这是与“平均Uplift”方法的核心区别
    cumulative_uplift = (reward_treat_cumsum / n_treat_cumsum_safe - reward_ctrl_cumsum / n_ctrl_cumsum_safe) * (n_treat_cumsum + n_ctrl_cumsum)

    # Step 4: 归一化坐标轴
    population_fraction = np.arange(1, n_total + 1) / n_total
    x_coords = [0] + population_fraction.tolist()
    
    # Y轴通过除以总Uplift进行归一化
    total_uplift = cumulative_uplift.iloc[-1]
    if total_uplift != 0:
        y_coords_normalized = (cumulative_uplift / total_uplift).tolist()
    else:
        y_coords_normalized = [0] * n_total
    y_coords = [0] + y_coords_normalized
    
    # Step 5: 计算AUUC (曲线下面积) 和基线AUUC
    auuc_score = np.trapz(y=y_coords, x=x_coords)
    # 在归一化坐标系下，随机基线是y=x的对角线，其面积固定为0.5
    baseline_auuc = 0.5
    
    # Step 6: 计算Qini系数
    qini_coefficient = 2 * (auuc_score - baseline_auuc)
    
    # Step 7: 创建结果字典
    results = {
        'auuc_score': float(auuc_score),
        'baseline_auuc': float(baseline_auuc),
        'qini_coefficient': float(qini_coefficient),
    }
    
    # # Step 8: 绘制图表
    # plt.figure(figsize=(12, 7))
    
    # # 绘制AUUC曲线
    # plt.plot(x_coords, y_coords, 'b-', linewidth=2.5, label=f'Uplift Model (AUUC = {auuc_score:.4f})')
    
    # # 绘制随机基线（对角线）
    # plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.7, label='Random Baseline (AUUC = 0.5)')
    
    # # 绘制零线
    # plt.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    # # 填充曲线下面积
    # plt.fill_between(x_coords, y_coords, 0, alpha=0.2, color='blue')
    
    # # 设置图表属性
    # # plt.xlabel('Population Fraction (Sorted by Uplift Score)', fontsize=12)
    # plt.xlabel('The Count of Smaples', fontsize=12)
    # # plt.ylabel('Normalized Cumulative Uplift', fontsize=12)
    # plt.ylabel('Incremental Reward', fontsize=12)
    
    # title = f'{uplift_col} - AUUC Curve - {reward_col}'
    # plt.title(title, fontsize=14, fontweight='bold')
    
    # # 添加AUUC和Qini系数信息
    # info_text = f'AUUC Score: {auuc_score:.4f}\nQini Coefficient: {qini_coefficient:.4f}\nTotal Uplift: {total_uplift:.2f}\nSample Size: {n_total}'
    # plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
    #          fontsize=10, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # plt.grid(True, alpha=0.3)
    # plt.legend(loc='upper left')
    # plt.xlim(0, 1)
    # plt.ylim(min(y_coords) - 0.05, max(y_coords) + 0.05)
    
    # plt.tight_layout()
    # # plt.show()
    # current_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # plot_save_path = f"{model_path}/result/auuc_{reward_col}_{uplift_col}_{current_ts}.pdf"
    # plt.savefig(plot_save_path)

    # plt.close()


    # Step 8: 绘制图表（仅调整风格，不改任何计算逻辑）
    fig, ax = plt.subplots(figsize=(8, 6))  # 更接近示例图比例

    # 主曲线：线更粗、无 marker
    ax.plot(
        x_coords, y_coords,
        linewidth=2.8,
        label=f'CC-DFL(Ours) (AUUC = {auuc_score:.4f})'
    )

    # Random baseline：黑色虚线对角线（示例图风格）
    ax.plot(
        [0, 1], [0, 1],
        linestyle='--',
        linewidth=2.5,
        color='black',
        label='Random'
    )

    # 网格：淡一点
    ax.grid(True, alpha=0.25)

    # 轴标签（不改含义；你原本就是归一化 x/y）
    # 如果你希望完全沿用你原来的文字，把下面两行改回你原来的 xlabel/ylabel 即可
    ax.set_xlabel('The Count of Samples', fontsize=18)
    ax.set_ylabel('Incremental Reward(ΔR)', fontsize=18)

    # 刻度字体
    ax.tick_params(axis='both', labelsize=14)

    # # 范围：归一化坐标
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    # 图例：右下角、带框（示例图风格）
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=16)

    # 去掉标题/文本框/填充（示例图没有这些）
    # title = f'{uplift_col} - AUUC Curve - {reward_col}'
    # ax.set_title(title, fontsize=14, fontweight='bold')

    fig.tight_layout()

    current_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_save_path = f"{model_path}/result/auuc_{reward_col}_{uplift_col}_{current_ts}.pdf"
    fig.savefig(plot_save_path)
    plt.close(fig)

    print(results)



# ## Uplift Bar Plot

# In[14]:


def calculate_and_plot_uplift_bar(df, target_col='paid', treatment_col='treatment', uplift_col='uplift', bins=20, model_path='unknown_model'):
    """
    计算并绘制 Uplift Bar Plot。

    该函数将用户按预测的 uplift 分数分箱，然后计算每个分箱内的真实平均 uplift，
    并将其可视化为柱状图，以评估模型的排序能力。

    Args:
        df (pd.DataFrame): 包含评估数据的 DataFrame。
        target_col (str): 结果/收益列的名称 (例如 'gmv')。
        treatment_col (str): 区分实验组和对照组的列名。
        uplift_col (str): 模型预测的 uplift 分数列名。
        bins (int): 分箱数量，默认为10（十分位）。
        model_path (str): 模型路径，用于生成图像文件名和标题。
    """
    # 1. 按模型预测的 uplift 分数降序排列
    df_sorted = df.sort_values(uplift_col, ascending=True)

    # 2. 使用 pd.qcut 创建分箱，确保每个分箱样本量大致相等
    try:
        df_sorted['bin'] = pd.qcut(df_sorted[uplift_col], q=bins, labels=False, duplicates='drop')
        # 将分箱标签从 0-9 调整为 1-10，更直观
        df_sorted['bin'] = df_sorted['bin'] + 1
    except ValueError:
        print(f"警告: 无法创建 {bins} 个唯一的箱。可能是因为 uplift 分数分布过于集中。将减少箱数。")
        # 如果无法创建10个箱（例如，大量用户uplift分数相同），则减少箱数
        df_sorted['bin'] = pd.qcut(df_sorted[uplift_col], q=min(bins, 5), labels=False, duplicates='drop')
        df_sorted['bin'] = df_sorted['bin'] + 1
        bins = df_sorted['bin'].nunique()


    # 3. 按分箱进行分组，并计算每个分箱的真实 uplift
    actual_uplifts_per_bin = []
    predicted_uplifts_per_bin = []
    grouped = df_sorted.groupby('bin')

    for bin_name, group in grouped:
        treat_mask = group[treatment_col] == 1
        ctrl_mask = group[treatment_col] == 0

        # 计算每个分箱中实验组和对照组的平均收益
        mean_reward_treat = group.loc[treat_mask, target_col].mean() if treat_mask.sum() > 0 else 0
        mean_reward_ctrl = group.loc[ctrl_mask, target_col].mean() if ctrl_mask.sum() > 0 else 0
        
        # 计算真实 uplift
        actual_uplift = mean_reward_treat - mean_reward_ctrl
        actual_uplifts_per_bin.append(actual_uplift)

        # 计算该分箱的平均预测uplift
        predicted_uplift = group[uplift_col].mean()
        predicted_uplifts_per_bin.append(predicted_uplift)

    # 4. 绘制柱状图
    bin_labels = [f'Top {i*100/bins:.0f}-{(i+1)*100/bins:.0f}%' for i in range(bins)]
    
    # plt.figure(figsize=(12, 7))
    plt.figure(figsize=(8, 6))
    x = np.arange(len(bin_labels))
    num_actual_bins = len(actual_uplifts_per_bin)
    x = np.arange(num_actual_bins)
    
    width = 0.35
    FONT_SIZE = 16 

    bars1 = plt.bar(x - width/2, actual_uplifts_per_bin, width, color='darkblue', label='True Uplift')
    bars2 = plt.bar(x + width/2, predicted_uplifts_per_bin, width, color='orange', label='Predicted Uplift')
    
    # 在每个柱子上方显示数值
    # for bar in bars1:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=FONT_SIZE)

    # for bar in bars2:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=FONT_SIZE)

    # 绘制一条代表整体平均Uplift的基准线
    overall_average_uplift = (df.loc[df[treatment_col] == 1, target_col].mean() - 
                              df.loc[df[treatment_col] == 0, target_col].mean())
    plt.axhline(y=overall_average_uplift, color='r', linestyle='--', label=f'Overall Avg Uplift ({overall_average_uplift:.4f})')

    # plt.title(f'Uplift Bar Plot for Model: {model_path}', fontsize=FONT_SIZE)
    plt.xlabel('User Deciles (Sorted by Predicted Uplift)', fontsize=FONT_SIZE)
    plt.ylabel(f'Average Uplift ({target_col})', fontsize=FONT_SIZE)
    plt.xticks(x, bin_labels, rotation=45, ha='right', fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 5. 保存图像
    # 清理模型路径以创建有效的文件名
    current_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_save_path = f"{model_path}/result/uplift_bar_{target_col}_{current_ts}.pdf"
    plt.savefig(plot_save_path)

    plt.close()  # 关闭图形，释放内存
    
    print(f"Uplift Bar Plot 已保存至: {plot_save_path}")


# In[16]:

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
    
    
    print("开始分批次进行预测...")
    max_batches_for_eval = config["max_batches_for_eval"]  # 示例：限制为100个批次

    # 遍历评估数据集的每个批次
    for i, (features_batch, labels_batch) in enumerate(eval_samples):
        # 检查是否达到批次数限制
        if max_batches_for_eval is not None and i >= max_batches_for_eval:
            print(f"已达到最大评估批次数 {max_batches_for_eval}，停止预测。")
            break
        # 在当前批次上进行预测
        pred_outputs = model.predict(features_batch)

        # 1-step CCB: 直接使用 action=1 的预测概率作为 ranking score
        pred_prob_action_1 = pred_outputs['prob_action_1']
        pred_prob_action_0 = pred_outputs['prob_action_0']

        # 如果返回的是 Tensor，这里转 numpy
        if tf.is_tensor(pred_prob_action_1):
            pred_prob_action_1 = pred_prob_action_1.numpy()
        if tf.is_tensor(pred_prob_action_0):
            pred_prob_action_0 = pred_prob_action_0.numpy()

        # ranking score
        integrated_uplift_per_sample = pred_prob_action_1

        # 为了兼容你现有后续 DataFrame 字段名，先占位
        num_samples = len(pred_prob_action_1)
        pred_paid_uplift = np.zeros(num_samples, dtype=np.float32)
        pred_cost_uplift = np.zeros(num_samples, dtype=np.float32)
        roi = np.zeros(num_samples, dtype=np.float32)

        # treat/ctrl 预测头也不再有，统一占位
        pred_paid_treat = np.zeros(num_samples, dtype=np.float32)
        pred_cost_treat = np.zeros(num_samples, dtype=np.float32)
        pred_paid_ctrl = np.zeros(num_samples, dtype=np.float32)
        pred_cost_ctrl = np.zeros(num_samples, dtype=np.float32)

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

        # 1-step CCB 的真正 ranking score
        'uplift': final_uplifts,
        'action1_prob': final_uplifts,

        # 以下字段仅为兼容你现有绘图/日志逻辑，占位
        'roi': final_rois,
        'treat_paid': final_treat_paid,
        'treat_cost': final_treat_cost,
        'ctrl_paid': final_ctrl_paid,
        'ctrl_cost': final_ctrl_cost,
        'uplift_paid': final_uplift_paid,
        'uplift_cost': final_uplift_cost,
    })
    
    with tee_output(f"{model_path}/eval.log", mode="a", encoding="utf-8"):
        # 打印结果DataFrame的前几行以供查验
        print("\n评估结果DataFrame示例:")
        print(eval_df.head())
        eval_df['treatment'] = eval_df['treatment'].astype(int)
        
        # 7. 计算 AUCC 并获取绘图数据
        print("正在计算 AUCC 指标...")
        aucc_score = strict_aucc_algorithm2(df=eval_df)
        print(f"模型 {model_path} 的 AUCC 分数为: {aucc_score:.6f}")
        aucc_score_2 = calculate_and_save_aucc(df=eval_df)
        print(f"模型 {model_path} 的 AUCC公司版本 分数为: {aucc_score_2:.6f}")

        print("正在生成 AUUC Plot (Action Probability Ranking on paid)...")
        plot_auuc(df=eval_df, reward_col='paid', treatment_col='treatment', uplift_col='uplift')

        print("正在生成 AUUC Plot (Action Probability Ranking on cost)...")
        plot_auuc(df=eval_df, reward_col='cost', treatment_col='treatment', uplift_col='uplift')

        # plot_auuc(df=eval_df, reward_col='cost', treatment_col='treatment', uplift_col='uplift_cost')
        # # print(f"模型 {model_path} 的 基线AUUC 分数为: {baseline_auuc:.6f}, cost-uplift AUUC 分数为: {auuc:.6f}")
        # plot_auuc(df=eval_df,reward_col='paid', treatment_col='treatment', uplift_col='uplift_paid')

        # print(f"模型 {model_path} 的 基线AUUC 分数为: {baseline_auuc:.6f}, paid-uplift AUUC 分数为: {auuc:.6f}")
        # plot_auuc(df=eval_df, reward_col='cost', treatment_col='treatment', uplift_col='roi')
        # print(f"模型 {model_path} 的 基线AUUC 分数为: {baseline_auuc:.6f}, cost-roi AUUC 分数为: {auuc:.6f}")
        # plot_auuc(df=eval_df,reward_col='paid', treatment_col='treatment', uplift_col='roi')
        # print(f"模型 {model_path} 的 基线AUUC 分数为: {baseline_auuc:.6f}, paid-roi AUUC 分数为: {auuc:.6f}")
        
        # --- 新增：调用 Uplift Bar Plot 函数 ---
        # print("正在生成 Paid Uplift Bar Plot...")
        # calculate_and_plot_uplift_bar(df=eval_df, target_col='paid', uplift_col='uplift_paid', model_path=model_path, bins=10)
        
        # print("正在生成 Cost Uplift Bar Plot...")
        # calculate_and_plot_uplift_bar(df=eval_df, target_col='cost', uplift_col='uplift_cost', model_path=model_path, bins=10)
        
        print("正在生成 AUCC Plot (Uplift)...")
        get_aucc_plot(eval_df, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='uplift', treatment_index=1, model_path=model_path)
        
        print("正在生成 AUCC Plot (ROI)...")
        get_aucc_plot(eval_df, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='roi', treatment_index=1, model_path=model_path)
        


# In[ ]:

import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

def plot_aucc_from_json_v1(json_path: str, plot_path: str = 'aucc_comparison.png', model_names: Optional[List[str]] = None):
    """
    从 JSON 文件加载一个或多个模型的 AUCC 数据并绘制对比图。

    Args:
        json_path (str): 包含 AUCC 数据的 JSON 文件路径。
                         文件格式应为: { "model_name": {"aucc_score": float, "x_coords": list, "y_coords": list}, ... }
        plot_path (str, optional): 生成的图像保存路径. Defaults to 'aucc_comparison.png'.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results: Dict[str, Dict[str, Any]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取或解析文件 {json_path} 时出错: {e}")
        return

    if not all_results:
        print("JSON 文件为空或格式不正确，无法绘图。")
        return
    
    results_to_plot = all_results
    if model_names:
        # 过滤出指定要绘制的模型
        results_to_plot = {name: data for name, data in all_results.items() if name in model_names}
        
        # 提醒用户哪些指定的模型未找到
        found_models = set(results_to_plot.keys())
        not_found_models = set(model_names) - found_models
        if not_found_models:
            print(f"警告: 在 {json_path} 中未找到以下模型: {list(not_found_models)}")

    if not results_to_plot:
        print("没有找到可供绘制的模型数据。")
        return

    plt.figure(figsize=(10, 8))

    # 绘制每个模型的 AUCC 曲线
    for model_name, data in results_to_plot.items():
        if 'x_coords' in data and 'y_coords' in data and 'aucc_score' in data:
        # if (model_name in model_paths_else or model_name in model_paths_DFCL) and 'x_coords' in data and 'y_coords' in data and 'aucc_score' in data:
            plt.plot(data['x_coords'], data['y_coords'], marker='.', linestyle='-', label=f'{model_name} (AUCC = {data["aucc_score"]:.4f})')
        # elif (model_name in model_paths_else or model_name in model_paths_DFCL):
        #     # print(f"模型 '{model_name}' 的数据不完整，跳过绘图。")
        #     pass
        else:
            print(f"模型 '{model_name}' 的数据不完整，跳过绘图。")

    # 绘制随机线 (使用第一个模型的数据作为基准)
    # 假设所有模型的最终 ΔC 和 ΔR 相同
    first_model_data = next(iter(results_to_plot.values()))
    if 'x_coords' in first_model_data and 'y_coords' in first_model_data:
        max_delta_c = first_model_data['x_coords'][-1]
        max_delta_r = first_model_data['y_coords'][-1]
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', label='Random')

    plt.title('AUCC Curve Comparison')
    plt.xlabel('Cumulative Cost Difference (ΔC)')
    plt.ylabel('Cumulative Reward Difference (ΔR)')
    plt.legend()
    plt.grid(True)

    # 保存图像并关闭绘图窗口
    plt.savefig(plot_path)
    plt.close()
    print(f"AUCC 曲线对比图已保存至: {plot_path}")


def plot_aucc_from_json_v2(
    json_path: str,
    plot_path: str = 'aucc_comparison.png',
    model_names: Optional[List[str]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    fallback_to_basename: bool = True,
):
    """
    从 JSON 文件加载一个或多个模型的 AUCC 数据并绘制对比图。

    Args:
        json_path: AUCC 数据 JSON 路径
        plot_path: 输出图片路径
        model_names: 只绘制这些 key（通常是 model_path 列表）
        model_name_map: {model_path: display_name}，用于图例显示
        fallback_to_basename: 映射缺失时是否用 Path(model_path).name 回退
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results: Dict[str, Dict[str, Any]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取或解析文件 {json_path} 时出错: {e}")
        return

    if not all_results:
        print("JSON 文件为空或格式不正确，无法绘图。")
        return

    results_to_plot = all_results
    if model_names:
        results_to_plot = {k: v for k, v in all_results.items() if k in model_names}

        found = set(results_to_plot.keys())
        not_found = set(model_names) - found
        if not_found:
            print(f"警告: 在 {json_path} 中未找到以下模型: {list(not_found)}")

    if not results_to_plot:
        print("没有找到可供绘制的模型数据。")
        return

    plt.figure(figsize=(10, 8))

    model_name_map = model_name_map or {}

    for model_key, data in results_to_plot.items():
        if not ('x_coords' in data and 'y_coords' in data and 'aucc_score' in data):
            print(f"模型 '{model_key}' 的数据不完整，跳过绘图。")
            continue

        # --- 核心：图例显示名逻辑 ---
        display_name = model_name_map.get(model_key)
        if not display_name:
            display_name = Path(model_key).name if fallback_to_basename else model_key

        plt.plot(
            data['x_coords'],
            data['y_coords'],
            marker='.',
            linestyle='-',
            label=f'{display_name} (AUCC = {data["aucc_score"]:.4f})'
        )

    # 随机线：归一化坐标下就是 y=x
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', label='Random')

    plt.title('AUCC Curve Comparison')
    plt.xlabel('Cumulative Cost Difference (ΔC)')
    plt.ylabel('Cumulative Reward Difference (ΔR)')
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_path)
    plt.close()
    print(f"AUCC 曲线对比图已保存至: {plot_path}")

from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

def plot_aucc_from_json_v3(
    json_path: str,
    plot_path: str = 'aucc_comparison.png',
    model_names: Optional[List[str]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    fallback_to_basename: bool = True,
):
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

    # ✅ 核心：按 model_names 的顺序决定绘图顺序（也决定 legend 顺序）
    if model_names:
        ordered_keys = [k for k in model_names if k in all_results]
        not_found = [k for k in model_names if k not in all_results]
        if not_found:
            print(f"警告: 在 {json_path} 中未找到以下模型: {not_found}")
    else:
        # 不传 model_names 时，就按 JSON 的 key 顺序画
        ordered_keys = list(all_results.keys())

    if not ordered_keys:
        print("没有找到可供绘制的模型数据。")
        return

    plt.figure(figsize=(10, 8))

    for model_key in ordered_keys:
        data = all_results.get(model_key, {})
        if not ('x_coords' in data and 'y_coords' in data and 'aucc_score' in data):
            print(f"模型 '{model_key}' 的数据不完整，跳过绘图。")
            continue

        display_name = model_name_map.get(model_key)
        if not display_name:
            display_name = Path(model_key).name if fallback_to_basename else model_key

        plt.plot(
            data['x_coords'],
            data['y_coords'],
            marker='.',
            linestyle='-',
            label=f'{display_name} (AUCC = {data["aucc_score"]:.4f})'
        )

    # 随机线放最后（legend 也会在最后）
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', label='Random')

    plt.title('AUCC Curve Comparison')
    plt.xlabel('Cumulative Cost Difference (ΔC)')
    plt.ylabel('Cumulative Reward Difference (ΔR)')
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_path)
    plt.close()
    print(f"AUCC 曲线对比图已保存至: {plot_path}")

def plot_aucc_from_json_v4(
    json_path: str,
    plot_path: str = 'aucc_comparison.png',
    model_names: Optional[List[str]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    fallback_to_basename: bool = True,
):
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

    # ✅ 核心：按 model_names 的顺序决定绘图顺序（也决定 legend 顺序）
    if model_names:
        ordered_keys = [k for k in model_names if k in all_results]
        not_found = [k for k in model_names if k not in all_results]
        if not_found:
            print(f"警告: 在 {json_path} 中未找到以下模型: {not_found}")
    else:
        # 不传 model_names 时，就按 JSON 的 key 顺序画
        ordered_keys = list(all_results.keys())

    if not ordered_keys:
        print("没有找到可供绘制的模型数据。")
        return

    OTHER_COLORS = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
    ]
    color_idx = 0  # 给非 ours 曲线依次分配颜色

    base_lw = 2.0  # 其他模型的线宽（你也可以调）
    highlight_name = "CC-DFL(Ours)"  # 这里改成你图例里真正显示的名字

    for model_key in ordered_keys:
        data = all_results.get(model_key, {})
        if not ('x_coords' in data and 'y_coords' in data and 'aucc_score' in data):
            print(f"模型 '{model_key}' 的数据不完整，跳过绘图。")
            continue

        # 图例显示名
        display_name = model_name_map.get(model_key)
        if not display_name:
            display_name = Path(model_key).name if fallback_to_basename else model_key

        # --- 核心：对 CC-DFL(Ours) 特殊处理 ---
        is_highlight = (display_name == highlight_name)

        if is_highlight:
            color = "red"
            lw = base_lw * 1.2
            zorder = 10
        else:
            color = OTHER_COLORS[color_idx % len(OTHER_COLORS)]
            color_idx += 1
            lw = base_lw
            zorder = 1

        plt.plot(
            data['x_coords'],
            data['y_coords'],
            linestyle='-',
            linewidth=lw,
            color=color,
            marker=None,        # 已去掉 marker
            zorder=zorder,
            label=f'{display_name} (AUCC = {data["aucc_score"]:.4f})'
        )

    # 随机线：归一化坐标下就是 y=x
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', label='Random')

    plt.title('AUCC Curve Comparison')
    plt.xlabel('Cumulative Cost Difference (ΔC)')
    plt.ylabel('Cumulative Reward Difference (ΔR)')
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"AUCC 曲线对比图已保存至: {plot_path}")

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
    from typing import Any, Dict, List, Optional

    # ====== ✅ 统一控制字体大小（比原 legend 更大） ======
    legend_fs = 16          # 图例字号（你觉得还不够就继续加到 18/20）
    axis_label_fs = legend_fs   # 横纵轴文字和图例一样大
    tick_fs = legend_fs        # 刻度数字也调到同一大小
    # ================================================

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

    # ✅ 按 model_names 的顺序决定绘图顺序（也决定 legend 顺序）
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

    # （可选）PDF 导出时常用设置：字体嵌入，避免投稿时字体替换
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # （可选）全局字体大小也一并设置（避免局部漏掉）
    plt.rcParams.update({
        "font.size": legend_fs,
        "axes.labelsize": axis_label_fs,
        "xtick.labelsize": tick_fs,
        "ytick.labelsize": tick_fs,
        "legend.fontsize": legend_fs,
    })

    OTHER_COLORS = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
    ]
    color_idx = 0

    base_lw = 2.5
    highlight_name = "CC-DFL"  # 这里改成你图例里真正显示的名字

    # ✅ 建议配合字号变大稍微放大画布
    plt.figure(figsize=(7.2, 5.4))

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
            lw = base_lw * 1.2
            zorder = 10
        else:
            color = OTHER_COLORS[color_idx % len(OTHER_COLORS)]
            color_idx += 1
            lw = base_lw
            zorder = 1

        plt.plot(
            data['x_coords'],
            data['y_coords'],
            linestyle='-',
            linewidth=lw,
            color=color,
            marker=None,
            zorder=zorder,
            label=f'{display_name} ({data["aucc_score"]:.4f})',
        )

    # 随机线：归一化坐标下就是 y=x
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=base_lw, label='Random', alpha=0.3)

    # ✅ 去掉标题（不再设置 plt.title）
    # plt.title('AUCC Curve Comparison')  # 删除/注释

    # ✅ 横纵轴文字与图例一样大（由 rcParams 控制；这里写也行）
    plt.xlabel('Incremental Cost(ΔC)', fontsize=axis_label_fs)
    plt.ylabel('Incremental Reward(ΔR)', fontsize=axis_label_fs)

    # ✅ tick 字号与图例一致
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)

    # ✅ 图例字号更大，并可调位置/列数避免遮挡
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


# In[22]:


import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

def plot_auuc_from_json(json_path: str, plot_path: str = 'auuc_comparison.png', model_names: Optional[List[str]] = None):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results: Dict[str, Dict[str, Any]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取或解析文件 {json_path} 时出错: {e}")
        return

    if not all_results:
        print("JSON 文件为空或格式不正确，无法绘图。")
        return
    
    results_to_plot = all_results
    if model_names:
        # 过滤出指定要绘制的模型
        results_to_plot = {name: data for name, data in all_results.items() if name in model_names}
        
        # 提醒用户哪些指定的模型未找到
        found_models = set(results_to_plot.keys())
        not_found_models = set(model_names) - found_models
        if not_found_models:
            print(f"警告: 在 {json_path} 中未找到以下模型: {list(not_found_models)}")

    if not results_to_plot:
        print("没有找到可供绘制的模型数据。")
        return

    plt.figure(figsize=(10, 8))

    # 绘制每个模型的 AUUC 曲线
    for model_name, data in results_to_plot.items():
        if 'x_coords' in data and 'y_coords' in data and 'auuc_score' in data:
            plt.plot(data['x_coords'], data['y_coords'], marker='.', linestyle='-', label=f'{model_name} (AUUC = {data["auuc_score"]:.4f})')
        else:
            print(f"模型 '{model_name}' 的数据不完整，跳过绘图。")

    # 绘制随机基线
    # 由于坐标已归一化，随机基线是一条从(0,0)到(1,1)的对角线
    first_model_data = next(iter(results_to_plot.values()))
    if 'baseline_auuc' in first_model_data:
        baseline_score = first_model_data['baseline_auuc']
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', 
                 label=f'Random Baseline (AUUC = {baseline_score:.4f})')

    plt.title('Normalized Cumulative Uplift Curve (AUUC)')
    plt.xlabel('Population Fraction')
    plt.ylabel('Normalized Cumulative Uplift')
    plt.legend()
    plt.grid(True)

    # 保存图像并关闭绘图窗口
    plt.savefig(plot_path)
    plt.close()
    print(f"AUUC 曲线对比图已保存至: {plot_path}")
    
json_file_path_auuc = auuc_save_path
output_image_path_auuc = f'result/auuc_curves_{current_time}.png'

# 调用函数生成图像
# plot_auuc_from_json(json_file_path_auuc, output_image_path_auuc, model_names = model_paths_DFCL + model_paths_else)