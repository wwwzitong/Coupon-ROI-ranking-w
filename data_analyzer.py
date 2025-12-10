# import pandas as pd
# import os


# def analyze_labels_and_treatment_from_csv(file_path):
#     """
#     使用 Pandas 从 CSV 文件分析标签的稀疏度和按 treatment 分组的效果。
#     """
#     print("开始分析标签和优惠策略效果...")
    
#     # 为了性能，只读取分析所需的列
#     required_cols = ["conversion", "visit",  "treatment"]
#     try:
#         # 使用 Pandas 读取 CSV 文件
#         df = pd.read_csv(file_path, usecols=required_cols)
#     except FileNotFoundError:
#         print(f"错误: 文件未找到于 {file_path}")
#         return []
#     except ValueError as e:
#         # 当 usecols 中指定的列在 CSV 中不存在时，会触发此错误
#         print(f"错误: CSV文件中缺少必要的列。 {e}")
#         return []

#     report_lines = []
    
#     # --- 1. 全局标签分析 (Pandas 实现) ---
#     print("--> 正在执行全局标签分析...")
    
#     total_count = len(df)
    
#     report_lines.append("="*80)
#     report_lines.append("标签与优惠策略分析报告 (基于 Pandas)")
#     report_lines.append("="*80)
#     report_lines.append("\n--- 1. 目标标签分析 ---")

#     if total_count > 0:
#         gmv_zero_count = (df['conversion'] == 0).sum()
#         cost_zero_count = (df['visit'] == 0).sum()
        
#         # 筛选非零值以计算平均值
#         gmv_non_zero = df.loc[df['conversion'] > 0, 'conversion']
#         cost_non_zero = df.loc[df['visit'] > 0, 'visit']

#         gmv_sparsity = gmv_zero_count / total_count
#         cost_sparsity = cost_zero_count / total_count
        
#         avg_gmv_non_zero = gmv_non_zero.mean() if not gmv_non_zero.empty else 0
#         avg_cost_non_zero = cost_non_zero.mean() if not cost_non_zero.empty else 0

#         report_lines.append(f"总分析样本数: {total_count}")
#         report_lines.append(f"GMV 为 0 的样本占比 (稀疏度): {gmv_sparsity:.2%}")
#         report_lines.append(f"Cost 为 0 的样本占比 (稀疏度): {cost_sparsity:.2%}")
#         report_lines.append(f"\n非零 GMV 样本的平均值: {avg_gmv_non_zero:.4f}")
#         report_lines.append(f"非零 Cost 样本的平均值: {avg_cost_non_zero:.4f}")
#     else:
#         report_lines.append("未分析任何样本。")

#     # --- 2. 按优惠策略分组分析 (Pandas 实现) ---
#     print("--> 正在执行按优惠策略分组分析...")
    
#     # 使用 Pandas 的 groupby 和 agg 功能
#     treatment_stats = df.groupby("treatment").agg(
#         sample_count=('treatment', 'size'),
#         avg_gmv=('conversion', 'mean'),
#         avg_cost=('visit', 'mean')
#     )

#     report_lines.append("\n--- 2. 优惠策略效果分析 ---")
#     header = f"{'Treatment':<12} | {'样本数':<12} | {'样本占比':<10} | {'平均GMV':<12} | {'平均Cost':<12}"
#     report_lines.append(header)
#     report_lines.append("-" * len(header))

#     # 迭代 groupby 结果
#     for t_idx, row in treatment_stats.iterrows():
#         count_in_group = row['sample_count']
#         proportion = count_in_group / total_count if total_count > 0 else 0
#         avg_gmv = row['avg_gmv']
#         avg_cost = row['avg_cost']
        
#         line = f"{t_idx:<12} | {count_in_group:<12} | {proportion:<10.2%} | {avg_gmv:<12.4f} | {avg_cost:<12.4f}"
#         report_lines.append(line)
    
#     return report_lines


# # --- 主执行逻辑 ---
# # !! 请将此路径替换为您的实际 CSV 文件路径 !!
# CSV_FILE_PATH = '/home/kongli/WangHe/data/criteo_data.csv' 

# # 执行分析
# report_lines = analyze_labels_and_treatment_from_csv(CSV_FILE_PATH)

# # --- 输出报告 ---
# if report_lines:
#     final_report_str = "\n".join(report_lines)
#     print("\n\n" + final_report_str)


import json

def extract_aucc_scores(file_path):
    """
    读取指定的 JSON 文件，并提取每个 model_path 及其对应的 aucc_score。

    Args:
        file_path (str): 存储 AUCC 结果的 JSON 文件路径。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查路径是否正确。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 的 JSON 格式不正确，无法解析。")
        return

    print("=" * 60)
    print("模型路径与 AUCC 分数对照表：")
    print("-" * 60)

    # 检查 all_results 是否为字典
    if not isinstance(all_results, dict):
        print("错误：JSON 文件的内容不是预期的字典格式。")
        return

    # 遍历字典中的每个条目
    for model_path, data in all_results.items():
        # 确保每个条目的值也是一个字典，并且包含 'aucc_score'
        if isinstance(data, dict) and 'aucc_score' in data:
            aucc_score = data['aucc_score']
            # 格式化输出，使路径和分数对齐
            print(f"{model_path:<45} AUCC Score: {aucc_score}")
        else:
            print(f"警告：跳过无效或格式不正确的条目 -> {model_path}")
    
    print("=" * 60)

# --- 使用方法 ---
if __name__ == "__main__":
    # 请将这里的 'path/to/your/aucc_results.json' 替换为你的实际文件路径
    aucc_json_path = '/home/kongli/roi_prediction_date1015/result1107_4/result_aucc.json' 
    
    extract_aucc_scores(aucc_json_path)