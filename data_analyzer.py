import pandas as pd
import os


def analyze_labels_and_treatment_from_csv(file_path):
    """
    使用 Pandas 从 CSV 文件分析标签的稀疏度和按 treatment 分组的效果，
    并额外分析 treatment=0 时 cost(visit) 的分布，判断是否存在极端大值。
    """
    print("开始分析标签和优惠策略效果...")
    
    # 为了性能，只读取分析所需的列
    required_cols = ["conversion", "visit", "treatment"]
    try:
        # 使用 Pandas 读取 CSV 文件
        df = pd.read_csv(file_path, usecols=required_cols)
    except FileNotFoundError:
        print(f"错误: 文件未找到于 {file_path}")
        return []
    except ValueError as e:
        # 当 usecols 中指定的列在 CSV 中不存在时，会触发此错误
        print(f"错误: CSV文件中缺少必要的列。 {e}")
        return []

    report_lines = []
    
    # --- 1. 全局标签分析 (Pandas 实现) ---
    print("--> 正在执行全局标签分析...")
    
    total_count = len(df)
    
    report_lines.append("=" * 80)
    report_lines.append("标签与优惠策略分析报告 (基于 Pandas)")
    report_lines.append("=" * 80)
    report_lines.append("\n--- 1. 目标标签分析 ---")

    if total_count > 0:
        gmv_zero_count = (df['conversion'] == 0).sum()
        cost_zero_count = (df['visit'] == 0).sum()
        
        # 筛选非零值以计算平均值
        gmv_non_zero = df.loc[df['conversion'] > 0, 'conversion']
        cost_non_zero = df.loc[df['visit'] > 0, 'visit']

        gmv_sparsity = gmv_zero_count / total_count
        cost_sparsity = cost_zero_count / total_count
        
        avg_gmv_non_zero = gmv_non_zero.mean() if not gmv_non_zero.empty else 0
        avg_cost_non_zero = cost_non_zero.mean() if not cost_non_zero.empty else 0

        report_lines.append(f"总分析样本数: {total_count}")
        report_lines.append(f"GMV 为 0 的样本占比 (稀疏度): {gmv_sparsity:.2%}")
        report_lines.append(f"Cost 为 0 的样本占比 (稀疏度): {cost_sparsity:.2%}")
        report_lines.append(f"\n非零 GMV 样本的平均值: {avg_gmv_non_zero:.4f}")
        report_lines.append(f"非零 Cost 样本的平均值: {avg_cost_non_zero:.4f}")
    else:
        report_lines.append("未分析任何样本。")

    # --- 2. 按优惠策略分组分析 (Pandas 实现) ---
    print("--> 正在执行按优惠策略分组分析...")
    
    # 使用 Pandas 的 groupby 和 agg 功能
    treatment_stats = df.groupby("treatment").agg(
        sample_count=('treatment', 'size'),
        avg_gmv=('conversion', 'mean'),
        avg_cost=('visit', 'mean')
    )

    report_lines.append("\n--- 2. 优惠策略效果分析 ---")
    header = f"{'Treatment':<12} | {'样本数':<12} | {'样本占比':<10} | {'平均GMV':<12} | {'平均Cost':<12}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    # 迭代 groupby 结果
    for t_idx, row in treatment_stats.iterrows():
        count_in_group = row['sample_count']
        proportion = count_in_group / total_count if total_count > 0 else 0
        avg_gmv = row['avg_gmv']
        avg_cost = row['avg_cost']
        
        line = f"{t_idx:<12} | {count_in_group:<12} | {proportion:<10.2%} | {avg_gmv:<12.4f} | {avg_cost:<12.4f}"
        report_lines.append(line)

    # --- 3. treatment=0 时 cost(visit) 分布分析 ---
    print("--> 正在执行 treatment=0 时 Cost 分布分析...")
    report_lines.append("\n--- 3. Treatment=0 时 Cost(visit) 分布分析 ---")

    df_t0 = df[df["treatment"] == 0]

    if len(df_t0) == 0:
        report_lines.append("没有 treatment=0 的样本。")
    else:
        t0_cost = df_t0["visit"].dropna()
        t0_cost_non_zero = t0_cost[t0_cost > 0]

        report_lines.append(f"treatment=0 样本数: {len(df_t0)}")
        report_lines.append(f"其中 cost=0 的样本数: {(t0_cost == 0).sum()}")
        report_lines.append(f"其中 cost>0 的样本数: {len(t0_cost_non_zero)}")

        if len(t0_cost) > 0:
            desc = t0_cost.describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.995, 0.999])

            report_lines.append("\n[全部 t=0 cost 的描述统计]")
            report_lines.append(f"mean   : {desc['mean']:.6f}")
            report_lines.append(f"std    : {desc['std']:.6f}" if 'std' in desc else "std    : NA")
            report_lines.append(f"min    : {desc['min']:.6f}")
            report_lines.append(f"50%    : {desc['50%']:.6f}")
            report_lines.append(f"90%    : {desc['90%']:.6f}")
            report_lines.append(f"95%    : {desc['95%']:.6f}")
            report_lines.append(f"99%    : {desc['99%']:.6f}")
            report_lines.append(f"99.5%  : {desc['99.5%']:.6f}")
            report_lines.append(f"99.9%  : {desc['99.9%']:.6f}")
            report_lines.append(f"max    : {desc['max']:.6f}")

            # IQR 异常值判定
            q1 = t0_cost.quantile(0.25)
            q3 = t0_cost.quantile(0.75)
            iqr = q3 - q1
            upper_bound_iqr = q3 + 1.5 * iqr
            extreme_iqr = t0_cost[t0_cost > upper_bound_iqr]

            # 用 99 分位做一个更直观的极端值参考
            q99 = t0_cost.quantile(0.99)
            extreme_q99 = t0_cost[t0_cost > q99]

            report_lines.append("\n[极端值检测]")
            report_lines.append(f"IQR 上界阈值: {upper_bound_iqr:.6f}")
            report_lines.append(f"超过 IQR 上界的样本数: {len(extreme_iqr)}")
            report_lines.append(f"超过 IQR 上界的样本占比: {len(extreme_iqr) / len(t0_cost):.2%}")

            if len(extreme_iqr) > 0:
                report_lines.append(f"IQR 异常值中的最大 cost: {extreme_iqr.max():.6f}")

            report_lines.append(f"99分位阈值: {q99:.6f}")
            report_lines.append(f"超过 99 分位的样本数: {len(extreme_q99)}")
            report_lines.append(f"超过 99 分位的样本占比: {len(extreme_q99) / len(t0_cost):.2%}")

        if len(t0_cost_non_zero) > 0:
            desc_nz = t0_cost_non_zero.describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.995, 0.999])

            report_lines.append("\n[仅对 t=0 且 cost>0 的描述统计]")
            report_lines.append(f"mean   : {desc_nz['mean']:.6f}")
            report_lines.append(f"std    : {desc_nz['std']:.6f}" if 'std' in desc_nz else "std    : NA")
            report_lines.append(f"min    : {desc_nz['min']:.6f}")
            report_lines.append(f"50%    : {desc_nz['50%']:.6f}")
            report_lines.append(f"90%    : {desc_nz['90%']:.6f}")
            report_lines.append(f"95%    : {desc_nz['95%']:.6f}")
            report_lines.append(f"99%    : {desc_nz['99%']:.6f}")
            report_lines.append(f"99.5%  : {desc_nz['99.5%']:.6f}")
            report_lines.append(f"99.9%  : {desc_nz['99.9%']:.6f}")
            report_lines.append(f"max    : {desc_nz['max']:.6f}")
        else:
            report_lines.append("treatment=0 下没有 cost>0 的样本。")
    
    return report_lines


# --- 主执行逻辑 ---
# !! 请将此路径替换为您的实际 CSV 文件路径 !!
CSV_FILE_PATH = './data/census1990_train.csv' 

# 执行分析
report_lines = analyze_labels_and_treatment_from_csv(CSV_FILE_PATH)

# --- 输出报告 ---
if report_lines:
    final_report_str = "\n".join(report_lines)
    print("\n\n" + final_report_str)