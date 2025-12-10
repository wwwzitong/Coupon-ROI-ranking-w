#!/usr/bin/env python
# coding: utf-8

# # Evaluation
# benchmark 包含DRM+DFL+DFCL*3

# In[1]:


from __future__ import print_function, absolute_import, division
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from data_utils import *                       #!!!!!!TODO:需要将model脚本转为py文件，正确import
from ecom_dfcl import EcomDFCL_v3
from ecom_drm import EcomDRM19
from ecom_dfl import EcomDFL
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有 GPU，自然不会加载 CUDA。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息（隐藏 INFO 和 WARNING）
DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']

# ### 上述代替import

# In[2]:


# --- 1. 配置字典（替代命令行参数） ---
config = {
    # 'eval_data': 'data/criteo_test.csv',
    'eval_data': 'data/criteo_train.csv',
    'batch_size': 1024*16,
    'max_batches_for_eval':79
}
# 步骤 3: 循环评估每个已保存的模型
model_paths_DFCL = [
    # "model/ecom_basemodel_2pos_lr4",
    # "model/ecom_basemodel_2pos",
    # "model/ecom_base_DFCL_2pll_2pos",
    # "model/ecom_base_DFCL_2pll_2pos_lr4",
    # "model/ecom_base_DFCL_3entropy_2pos_lr4",
    # "model/ecom_base_DFCL_3entropy_2pos",
    # "model/ecom_base_DFCL_4ifdl_2pos_lr4",
    # "model/ecom_base_DFCL_4ifdl_2pos",
    # "model/ecom_base_DFCL_2pll_2pos_gradient_lr4",
    # "model/ecom_base_DFCL_2pll_2pos_gradient",
    # "model/ecom_base_DFCL_3entropy_2pos_gradient",
    # "model/ecom_base_DFCL_3entropy_2pos_gradient_lr4",
    # "model/ecom_base_DFCL_4ifdl_2pos_gradient_lr4",
    # "model/ecom_base_DFCL_4ifdl_2pos_gradient",
    "model/ecom_DFCL_3entropy_2pos"
]
model_paths_else = [
#     "model/ecom_DFL",
#     "model/ecom_DRM",
#     "model/ecom_DRM_original",
#     "model/ecom_DRMv2"
]


# In[3]:


# --- 2. 加载测试集 ---
dataset = CSVData()
eval_samples = dataset.prepare_dataset(
    config['eval_data'], 
    phase='eval', 
    batch_size=config['batch_size'], # 一次性加载所有数据进行评估 =None
    shuffle=False
)
label_name_list = ['treatment','paid','cost']
drop_list = ['paid','cost']

# --- Step: 将 dataset 转换为 (features, labels) 格式 ---
def _to_features_labels(parsed_example):
    # 提取 features（从 feature_name_list 中）
    features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}
    # 构建 labels 字典，特别处理 treatment 的反转
    labels = {}
    for name in label_name_list:
        value = parsed_example[name]
        # if name == 'treatment':
        #     # 反转：0 -> 1, 1 -> 0
        #     reversed_value = tf.cast(1 - value, tf.int32)
        #     labels[name] = reversed_value
        # else:
        labels[name] = value

    return features, labels  # 返回 (features, labels) 其中 labels 是 dict
# --- 应用 map 转换 ---
eval_samples = eval_samples.map(
    _to_features_labels,
    num_parallel_calls=4
).prefetch(1)


# In[5]:


treatment_order = [1, 0] #处理组为15off，另一组是空白组
ratios = [i / 100.0 for i in range(5, 105, 5)]
aucc_save_path = "result/result_aucc.json" #保存好坐标点，以便后续画图
auuc_path = "result/auuc.json" #保存好坐标点，以便后续画图


# ## AUCC

# In[6]:

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

#公司另外一个版本的AUCC
def calculate_and_save_aucc(df, reward_col='paid', cost_col='cost', treatment_col='treatment', uplift_col='uplift', uplift_gmv_col='uplift_gmv', uplift_cost_col='uplift_cost', treatment_val=1, control_val=0, n_bins=100):
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


def get_aucc_plot(pdf, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='uplift', treatment_index=1):
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
    plt.title("Normalized AUCC Curve")
    plt.show()
    aucc = np.trapz(df['norm_delta_gain'], df['norm_delta_cost'])
    random_area = np.trapz(df['norm_delta_cost'], df['norm_delta_cost'])
    aucc_score = aucc/2/random_area
    aucc_dict[pred_roi_col]['score'] = aucc_score
    aucc_dict[pred_roi_col]['treatment'] = treatment_index
    aucc_dict[pred_roi_col]['random_score'] = 0.5
    print(aucc_dict)
    return aucc_dict

# ## AUUC

# In[7]:
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


# ## Uplift Bar Plot

# In[8]:

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
    
    plt.figure(figsize=(12, 7))
    x = np.arange(len(bin_labels))
    num_actual_bins = len(actual_uplifts_per_bin)
    x = np.arange(num_actual_bins)
    
    width = 0.35

    bars1 = plt.bar(x - width/2, actual_uplifts_per_bin, width, color='darkblue', label='True Uplift')
    bars2 = plt.bar(x + width/2, predicted_uplifts_per_bin, width, color='orange', label='Predicted Uplift')
    
    # 在每个柱子上方显示数值
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if yval >= 0 else 'top', ha='center')

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if yval >= 0 else 'top', ha='center')

    # 绘制一条代表整体平均Uplift的基准线
    overall_average_uplift = (df.loc[df[treatment_col] == 1, target_col].mean() - 
                              df.loc[df[treatment_col] == 0, target_col].mean())
    plt.axhline(y=overall_average_uplift, color='r', linestyle='--', label=f'Overall Avg Uplift ({overall_average_uplift:.4f})')

    plt.title(f'Uplift Bar Plot for Model: {model_path}')
    plt.xlabel('User Deciles (Sorted by Predicted Uplift)')
    plt.ylabel(f'Average Uplift ({target_col.upper()})')
    plt.xticks(x, bin_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 5. 保存图像
    # 清理模型路径以创建有效的文件名
    sanitized_model_name = model_path.replace('/', '_').replace('\\', '_').replace('.', '')
    plot_save_path = f"result/uplift_bar_{target_col}_{sanitized_model_name}.png"
    plt.savefig(plot_save_path)
    plt.close()  # 关闭图形，释放内存
    
    print(f"Uplift Bar Plot 已保存至: {plot_save_path}")

# In[11]:

# --- 评估流程 (已完善) 倾向于每一个model单独计算完metric，进行数据保留后统一画图（因为模型输出接口不一致）---
print("开始评估流程...")

for model_path in model_paths_DFCL: 
    print(f"\n{'='*20} 正在评估模型: {model_path} {'='*20}")
    
    # 2.1 加载模型并进行预测
    print("加载模型并进行预测...")
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # 假设模型是 Keras SavedModel 格式
        model = tf.keras.models.load_model(model_path, compile=False) 
        
    # 初始化用于存储所有批次结果的列表
    all_uplifts = []
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
        all_rois.append(roi)
        all_treat_paid.append(pred_paid_treat)
        all_treat_cost.append(pred_cost_treat)
        all_ctrl_paid.append(pred_paid_ctrl)
        all_ctrl_cost.append(pred_cost_ctrl)
        all_paid_labels.append(labels_batch['paid'].numpy())
        all_cost_labels.append(labels_batch['cost'].numpy())
        all_treatment_labels.append(labels_batch['treatment'].numpy())

    print("所有批次预测完成，正在整合结果...")
    # 将所有批次的结果（list of arrays）拼接成一个大的Numpy数组
    final_uplifts = np.concatenate(all_uplifts)
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
        'ctrl_cost': final_ctrl_cost
    })
    

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

    print("正在计算 AUUC 指标...")
    auuc, baseline_auuc = calculate_auuc(df=eval_df)
    print(f"模型 {model_path} 的 基线AUUC 分数为: {baseline_auuc:.6f}, AUUC 分数为: {auuc:.6f}")
    
    # --- 新增：调用 Uplift Bar Plot 函数 ---
    print("正在生成 PAID Uplift Bar Plot...")
    calculate_and_plot_uplift_bar(df=eval_df, target_col='paid', model_path=model_path)
    
    print("正在生成 Cost Uplift Bar Plot...")
    calculate_and_plot_uplift_bar(df=eval_df, target_col='cost', model_path=model_path)
    
    get_aucc_plot(eval_df, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='uplift', treatment_index=1)
    get_aucc_plot(eval_df, treatment_col='treatment', gain_col='paid', cost_col='cost', pred_roi_col='roi', treatment_index=1)
    
    
    
    # 1127Addition：
    # --- 新增评估逻辑 ---
    from sklearn.metrics import roc_auc_score

    print("\n" + "-"*10 + " 额外评估指标 " + "-"*10)
    # 筛选出实验组数据用于评估
    treatment_df = eval_df[eval_df['treatment'] == 1]

    if treatment_df.empty:
        print("实验组无数据，跳过额外评估。")
    else:
        # 1. 统计模型预估值平均值和真实值平均值对比
        print("\n模型预估值与真实值均值对比 (实验组):")
        avg_pred_paid = treatment_df['treat_paid'].mean()
        avg_true_paid = treatment_df['paid'].mean()
        print(f"  - Paid: 预估平均值 = {avg_pred_paid:.4f}, 真实平均值 = {avg_true_paid:.4f}")

        avg_pred_cost = treatment_df['treat_cost'].mean()
        avg_true_cost = treatment_df['cost'].mean()
        print(f"  - Cost: 预估平均值 = {avg_pred_cost:.4f}, 真实平均值 = {avg_true_cost:.4f}")

        # 2. 计算并展示 paid 和 cost 的 regAUC
        print("\n计算 Regression AUC (regAUC, 在实验组上):")

        def calculate_reg_auc(y_true, y_pred, label_name):
            # 检查真实值是否都一样，无法计算AUC
            if y_true.nunique() <= 1:
                print(f"  - {label_name}: 无法计算regAUC，因实验组中'{label_name.lower()}'真实值单一。")
                return

            # 将回归问题转化为二分类问题来计算AUC
            binary_true = (y_true > y_true.median()).astype(int)
            
            # 检查二分后的标签是否只有一个类别
            if len(np.unique(binary_true)) <= 1:
                print(f"  - {label_name}: 无法计算regAUC，因真实值中位数导致所有样本归于一类。")
                return
            
            reg_auc = roc_auc_score(binary_true, y_pred)
            print(f"  - {label_name} regAUC: {reg_auc:.4f}")

        calculate_reg_auc(treatment_df['paid'], treatment_df['treat_paid'], 'Paid')
        calculate_reg_auc(treatment_df['cost'], treatment_df['treat_cost'], 'Cost')
        
        
    # 筛选出对照组数据用于评估
    control_df = eval_df[eval_df['treatment'] == 0]

    if control_df.empty:
        print("对照组无数据，跳过额外评估。")
    else:
        # 1. 统计模型预估值平均值和真实值平均值对比
        print("\n模型预估值与真实值均值对比 (对照组):")
        avg_pred_paid = control_df['ctrl_paid'].mean()
        avg_true_paid = control_df['paid'].mean()
        print(f"  - Paid: 预估平均值 = {avg_pred_paid:.4f}, 真实平均值 = {avg_true_paid:.4f}")

        avg_pred_cost = control_df['ctrl_cost'].mean()
        avg_true_cost = control_df['cost'].mean()
        print(f"  - Cost: 预估平均值 = {avg_pred_cost:.4f}, 真实平均值 = {avg_true_cost:.4f}")

        # 2. 计算并展示 paid 和 cost 的 regAUC
        print("\n计算 Regression AUC (regAUC, 在对照组上):")

        def calculate_reg_auc(y_true, y_pred, label_name):
            # 检查真实值是否都一样，无法计算AUC
            if y_true.nunique() <= 1:
                print(f"  - {label_name}: 无法计算regAUC，因实验组中'{label_name.lower()}'真实值单一。")
                return

            # 将回归问题转化为二分类问题来计算AUC
            binary_true = (y_true > y_true.median()).astype(int)
            
            # 检查二分后的标签是否只有一个类别
            if len(np.unique(binary_true)) <= 1:
                print(f"  - {label_name}: 无法计算regAUC，因真实值中位数导致所有样本归于一类。")
                return
            
            reg_auc = roc_auc_score(binary_true, y_pred)
            print(f"  - {label_name} regAUC: {reg_auc:.4f}")

        calculate_reg_auc(control_df['paid'], control_df['ctrl_paid'], 'Paid')
        calculate_reg_auc(control_df['cost'], control_df['ctrl_cost'], 'Cost')
    # --- 评估逻辑结束 ---

# In[12]:


import json
import matplotlib.pyplot as plt
from typing import Dict, Any

def plot_aucc_from_json(json_path: str, plot_path: str = 'aucc_comparison.png'):
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

    plt.figure(figsize=(10, 8))

    # 绘制每个模型的 AUCC 曲线
    for model_name, data in all_results.items():
        if 'x_coords' in data and 'y_coords' in data and 'aucc_score' in data:
            plt.plot(data['x_coords'], data['y_coords'], marker='.', linestyle='-', label=f'{model_name} (AUCC = {data["aucc_score"]:.4f})')
        else:
            print(f"模型 '{model_name}' 的数据不完整，跳过绘图。")

    # 绘制随机线 (使用第一个模型的数据作为基准)
    # 假设所有模型的最终 ΔC 和 ΔR 相同
    first_model_data = next(iter(all_results.values()))
    if 'x_coords' in first_model_data and 'y_coords' in first_model_data:
        max_delta_c = first_model_data['x_coords'][-1]
        max_delta_r = first_model_data['y_coords'][-1]
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', label='Random')

    plt.title('AUCC Curve Comparison')
    plt.xlabel('Cumulative visit Difference (ΔC)')
    plt.ylabel('Cumulative Reward Difference (ΔR)')
    plt.legend()
    plt.grid(True)

    # 保存图像并关闭绘图窗口
    plt.savefig(plot_path)
    plt.close()
    print(f"AUCC 曲线对比图已保存至: {plot_path}")
    
#  'result_aucc.json'
json_file_path = 'result/result_aucc.json'
output_image_path = 'result/aucc_curves.png'

# 调用函数生成图像
plot_aucc_from_json(json_file_path, output_image_path)

#  'result_aucc_2.json'
json_file_path_2 = 'result/result_aucc_v2.json'
output_image_path_2 = 'result/aucc_curves_ByteDance.png'

# 调用函数生成图像
plot_aucc_from_json(json_file_path_2, output_image_path_2)


# In[13]:


import json
import matplotlib.pyplot as plt
from typing import Dict, Any

def plot_auuc_from_json(json_path: str, plot_path: str = 'auuc_comparison.png'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results: Dict[str, Dict[str, Any]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取或解析文件 {json_path} 时出错: {e}")
        return

    if not all_results:
        print("JSON 文件为空或格式不正确，无法绘图。")
        return

    plt.figure(figsize=(10, 8))

    # 绘制每个模型的 AUUC 曲线
    for model_name, data in all_results.items():
        if 'x_coords' in data and 'y_coords' in data and 'auuc_score' in data:
            plt.plot(data['x_coords'], data['y_coords'], marker='.', linestyle='-', label=f'{model_name} (AUUC = {data["auuc_score"]:.4f})')
        else:
            print(f"模型 '{model_name}' 的数据不完整，跳过绘图。")

    # 绘制随机基线
    # 由于坐标已归一化，随机基线是一条从(0,0)到(1,1)的对角线
    first_model_data = next(iter(all_results.values()))
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
    
json_file_path_auuc = 'result/auuc.json'
output_image_path_auuc = 'result/auuc_curves.png'

# 调用函数生成图像
plot_auuc_from_json(json_file_path_auuc, output_image_path_auuc)