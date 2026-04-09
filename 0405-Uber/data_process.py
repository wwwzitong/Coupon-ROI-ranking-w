import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# ------------------------------------------------------------------------------
# STEP 1 — 获取 US Census 1990 数据
# ------------------------------------------------------------------------------
# 拉取 UCI 数据集 id=116，即 US Census Data (1990)
us_census = fetch_ucirepo(id=116)

# 特征矩阵 + 元标签（如果有）
X_raw = us_census.data.features  # pandas DataFrame
y_raw = us_census.data.targets   # pandas DataFrame or array

print("Raw shape:", X_raw.shape)
print("Raw columns:", X_raw.columns.tolist())

# ------------------------------------------------------------------------------
# STEP 2 — 筛选特定样本作为论文里模拟实验的子人群
# ------------------------------------------------------------------------------

# 根据论文设置的过滤条件：

# 1) 仅保留有一个或多个孩子的人
# mask_children = X_raw['iFertil'] <= 2
mask_children = X_raw['iFertil'].notna() & (X_raw['iFertil'] <= 2) & (X_raw['iFertil'] > 0)

# 2) 仅保留出生在美国的人
mask_citizen = X_raw['iCitizen'].notna() & (X_raw['iCitizen'] == 0)

# 3) 年龄小于 50 岁（原始 dAge 是分箱的编码，要根据论文设定做比较）
mask_age_under50 = X_raw['dAge'].notna() & (X_raw['dAge'] < 5)

# 组合筛选
mask = mask_children & mask_citizen & mask_age_under50
# mask = mask_citizen & mask_age_under50
df_filtered = X_raw[mask].copy()

print("Filtered shape:", df_filtered.shape)

# ------------------------------------------------------------------------------
# STEP 3 — 定义论文里的 Treatment / Gain / Cost 变量
# ------------------------------------------------------------------------------

# 定义 treatment：高于工作时长的中位数 => 1 else 0
median_hours = df_filtered['dHours'].median()
df_filtered['treatment'] = (df_filtered['dHours'] > median_hours).astype(int)

# 定义 gain outcome（论文把收入 dIncome1 当成收益）
df_filtered['conversion'] = df_filtered['dIncome1'].astype(float)

# 定义 cost outcome（论文里 cost = - iFertil）
# df_filtered['visit'] = df_filtered['iFertil'].astype(float)
max_ifertil = df_filtered['iFertil'].astype(float).max()
print("Max iFertil:", max_ifertil)
# df_filtered['visit'] = (max_ifertil - df_filtered['iFertil'].astype(float)) * 0.1
df_filtered['visit'] = df_filtered['iFertil'].astype(float)

print("Median hours:", median_hours)
print(df_filtered[['treatment','conversion','visit']].head())

# ------------------------------------------------------------------------------
# STEP 4 — 保留特征集（论文里保留 46 维去除混杂变量）
# ------------------------------------------------------------------------------

# 列出要排除的混杂变量（论文里删掉特定字段）
vars_to_drop = [
    'caseid',
    'dIncome2',
    'dIncome3',
    'dIncome4',
    'dIncome5',
    'dIncome6',
    'dIncome7',
    'dIncome8',
    'iMarital',
    'dAge',
    'dAncstry1',
    'dAncstry2',
    'dHours',
    'dIncome1',
    'iFertil',

    'dHour89',
    'dWeek89',
    'iWork89',
    'iWorklwk',
    'iYearwrk',
    'dRearning',
    'dRpincome',
    'dPoverty',
]
# 注意：根据论文具体说明，需确认这些列在 df_filtered 中的真实名称

# 手动检查是否在当前 DataFrame 中
vars_exist = [v for v in vars_to_drop if v in df_filtered.columns]
df_features = df_filtered.drop(columns=vars_exist)

# 只重命名除 treatment / conversion / visit 外的其余 46 列
cols_to_keep = ['treatment', 'conversion', 'visit']
cols_to_rename = [c for c in df_features.columns if c not in cols_to_keep]

rename_map = {old: f"f{i}" for i, old in enumerate(cols_to_rename)}
df_features = df_features.rename(columns=rename_map)

print("Final feature dimension:", df_features.shape)

# ------------------------------------------------------------------------------
# STEP 5 — 构造训练/验证/测试划分
# ------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

train, temp = train_test_split(df_features, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)

# ------------------------------------------------------------------------------
# (可选) 保存到本地 CSV
# ------------------------------------------------------------------------------

train.to_csv("../data/census1990_train.csv", index=False)
val.to_csv("../data/census1990_val.csv", index=False)
test.to_csv("../data/census1990_test.csv", index=False)