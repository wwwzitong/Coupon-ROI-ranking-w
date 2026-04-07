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
    'auuc_save_path': "result/result_auuc.json", #保存好坐标点，以便后续画图

    # ===== 新增：决策边界评测配置 =====
    'lambda_star': 0.1,          # 全局阈值 λ*
    'boundary_ratio': 0.1,       # 取 |u*(x)| 最小的 10% 作为边界样本
    'boundary_eps': None,        # 若不为 None，则优先使用 |u*(x)| < eps
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
    "./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10",
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

# ==================== 决策边界评测逻辑 ====================
def sign_binary(x: np.ndarray) -> np.ndarray:
    """
    二值符号函数：
    x >= 0 -> 1
    x < 0  -> -1
    """
    return np.where(x >= 0, 1, -1)


def get_boundary_mask(u_star: np.ndarray, eps: Optional[float] = None, boundary_ratio: Optional[float] = None) -> np.ndarray:
    """
    定义边界样本：
    1) 若 eps 不为 None，则 |u*(x)| <= eps
    2) 否则直接按 |u*(x)| 排序，取最小的 boundary_ratio 比例样本
    """
    abs_u = np.abs(u_star)
    n = len(abs_u)

    if eps is not None:
        return abs_u <= eps

    if boundary_ratio is None:
        raise ValueError("eps 和 boundary_ratio 不能同时为 None。")

    if not (0 < boundary_ratio <= 1):
        raise ValueError("boundary_ratio 必须在 (0, 1] 之间。")

    k = max(1, int(n * boundary_ratio))
    sorted_idx = np.argsort(abs_u)[:k]

    mask = np.zeros(n, dtype=bool)
    mask[sorted_idx] = True
    return mask


def evaluate_boundary_metrics(
    df: pd.DataFrame,
    lambda_star: float,
    pred_utility_col: str = 'pred_utility_lambda',
    true_reward_col: str = 'paid',
    true_cost_col: str = 'cost',
    eps: Optional[float] = None,
    boundary_ratio: Optional[float] = 0.1,
):
    """
    实现图片中的评测逻辑：
    1. Oracle Utility: u*(x) = r - λ*c
    2. 边界样本：|u*(x)| < eps 或取最小 boundary_ratio 比例
    3. Argmax Inversion Rate：边界样本上 sign(û) != sign(u*)
    4. 局部 MSE：边界样本 / 安全样本上的 MSE

    说明：
    - 这里默认使用测试集中的 factual paid/cost 作为 true reward/cost
    - 预测效用使用 pred_utility_lambda 列
    """
    eval_df = df.copy()

    # 1) Oracle utility
    eval_df['oracle_utility'] = eval_df[true_reward_col] - lambda_star * eval_df[true_cost_col]

    # 2) 边界样本 / 安全样本
    boundary_mask = get_boundary_mask(
        u_star=eval_df['oracle_utility'].values,
        eps=eps,
        boundary_ratio=boundary_ratio
    )
    safe_mask = ~boundary_mask

    u_true = eval_df['oracle_utility'].values
    u_pred = eval_df[pred_utility_col].values

    # 3) Argmax Inversion Rate
    if boundary_mask.sum() > 0:
        inversion_rate = np.mean(
            sign_binary(u_pred[boundary_mask]) != sign_binary(u_true[boundary_mask])
        )
    else:
        inversion_rate = np.nan

    # 4) 局部 MSE
    mse_boundary = np.mean((u_pred[boundary_mask] - u_true[boundary_mask]) ** 2) if boundary_mask.sum() > 0 else np.nan
    mse_safe = np.mean((u_pred[safe_mask] - u_true[safe_mask]) ** 2) if safe_mask.sum() > 0 else np.nan
    mse_all = np.mean((u_pred - u_true) ** 2)

    metrics = {
        'lambda_star': float(lambda_star),
        'num_total': int(len(eval_df)),
        'num_boundary': int(boundary_mask.sum()),
        'num_safe': int(safe_mask.sum()),
        'boundary_ratio_actual': float(boundary_mask.mean()),
        'argmax_inversion_rate_boundary': float(inversion_rate) if not np.isnan(inversion_rate) else np.nan,
        'mse_boundary': float(mse_boundary) if not np.isnan(mse_boundary) else np.nan,
        'mse_safe': float(mse_safe) if not np.isnan(mse_safe) else np.nan,
        'mse_all': float(mse_all),
    }

    eval_df['oracle_utility'] = u_true
    eval_df['is_boundary'] = boundary_mask
    eval_df['is_safe'] = safe_mask

    return metrics, eval_df
# ==================== 决策边界评测逻辑结束 ====================

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
    lambda_star = config['lambda_star']
    pred_utility_lambda = np.where(
        final_treatment == 1,
        final_treat_paid - lambda_star * final_treat_cost,
        final_ctrl_paid - lambda_star * final_ctrl_cost
    )
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

        # ===== 新增：固定 λ* 下的预测效用 û(x) =====
        # û(x) = uplift_paid - λ* uplift_cost
        # 'pred_utility_lambda': final_uplift_paid - lambda_star * final_uplift_cost,
        'pred_utility_lambda': pred_utility_lambda,
    })
    
    with tee_output(f"{model_path}/eval.log", mode="a", encoding="utf-8"):
        # 打印结果DataFrame的前几行以供查验
        print("\n评估结果DataFrame示例:")
        print(eval_df.head())
        eval_df['treatment'] = eval_df['treatment'].astype(int)
        
        # ===== 新增：决策边界评测 =====
        print("\n" + "-"*10 + " 决策边界评测 " + "-"*10)

        boundary_metrics, eval_df = evaluate_boundary_metrics(
            df=eval_df,
            lambda_star=config['lambda_star'],
            pred_utility_col='pred_utility_lambda',
            true_reward_col='paid',
            true_cost_col='cost',
            eps=config.get('boundary_eps', None),
            boundary_ratio=config.get('boundary_ratio', 0.1),
        )

        print(f"lambda*: {boundary_metrics['lambda_star']:.6f}")
        print(f"总样本数: {boundary_metrics['num_total']}")
        print(f"边界样本数: {boundary_metrics['num_boundary']}")
        print(f"安全样本数: {boundary_metrics['num_safe']}")
        print(f"实际边界样本占比: {boundary_metrics['boundary_ratio_actual']:.6f}")
        print(f"Argmax Inversion Rate (Boundary): {boundary_metrics['argmax_inversion_rate_boundary']:.6f}")
        print(f"Local MSE (Boundary): {boundary_metrics['mse_boundary']:.6f}")
        print(f"Local MSE (Safe): {boundary_metrics['mse_safe']:.6f}")
        print(f"Global MSE: {boundary_metrics['mse_all']:.6f}")

        # 保存明细，便于后续分析
        boundary_detail_path = f"{model_path}/result/boundary_eval_detail_lambda_{config['lambda_star']}.csv"
        eval_df.to_csv(boundary_detail_path, index=False, encoding='utf-8-sig')
        print(f"边界评测明细已保存至: {boundary_detail_path}")

        # 保存 summary json
        boundary_summary_path = f"{model_path}/result/boundary_eval_summary_lambda_{config['lambda_star']}.json"
        with open(boundary_summary_path, 'w', encoding='utf-8') as f:
            json.dump(boundary_metrics, f, indent=4, ensure_ascii=False)
        print(f"边界评测汇总已保存至: {boundary_summary_path}")
        
        
        
