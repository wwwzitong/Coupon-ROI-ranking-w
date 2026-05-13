# construct_osrct_criteo.py
# -*- coding: utf-8 -*-
"""
将 Criteo Uplift RCT 数据按 OSRCT 构造成带混杂的非 RCT / observational 数据。

示例：
python construct_osrct_criteo.py \
  --input criteo-uplift-v2.1.csv \
  --outdir osrct_criteo_conversion \
  --target conversion \
  --alphas 0 0.5 1 2 4 \
  --bias-target-mean 0.5 \
  --clip-low 0.02 \
  --clip-high 0.98 \
  --seed 42

小样本调试：
python construct_osrct_criteo.py \
  --input criteo-uplift-v2.1.csv \
  --outdir debug_osrct \
  --target conversion \
  --sample-n 200000 \
  --alphas 0 1 2

如果希望 accepted observational data 中的最终 P(T=1|X) 接近你设计的 biasing propensity，
而不是仅直接使用 Algorithm 2 的 P(T_s=1|X)，加：
  --target-final-propensity

注意：
1. 训练 biasing function 用 outcome Y 是为了构造 benchmark，不是给被评估的 causal model 使用。
2. 输出数据中的 __p_ts1、__p_accept、__outcome_score 等列不要作为 causal model 的输入特征。
3. treatment 应使用 Criteo 的 `treatment`，不要用 `exposure` 作为随机 treatment。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# -----------------------------
# 基础工具
# -----------------------------

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """数值稳定 sigmoid。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def safe_alpha_name(alpha: float) -> str:
    return str(alpha).replace("-", "m").replace(".", "p")


def infer_feature_cols(columns: Iterable[str]) -> List[str]:
    """自动识别 f0, f1, ..., f11 这样的特征列。"""
    feats = []
    for c in columns:
        if len(c) >= 2 and c[0] == "f" and c[1:].isdigit():
            feats.append(c)
    feats = sorted(feats, key=lambda s: int(s[1:]))
    if not feats:
        raise ValueError("没有找到 f0, f1, ... 形式的 Criteo 特征列。")
    return feats


def read_criteo(
    path: str,
    target_col: str = "conversion",
    treatment_col: str = "treatment",
    nrows: int | None = None,
    sample_n: int | None = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """读取 Criteo CSV，并尽量用小 dtype 降低内存。"""
    header = pd.read_csv(path, nrows=0)
    all_cols = list(header.columns)
    feature_cols = infer_feature_cols(all_cols)

    required = [target_col, treatment_col]
    missing = [c for c in required if c not in all_cols]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")

    # 额外保留常用列，方便后续分析；不存在则跳过。
    extra_cols = [c for c in ["visit", "conversion", "treatment", "exposure"] if c in all_cols]
    usecols = sorted(set(feature_cols + extra_cols + required), key=lambda x: all_cols.index(x))

    dtype: Dict[str, object] = {c: "float32" for c in feature_cols}
    for c in ["visit", "conversion", "treatment", "exposure"]:
        if c in usecols:
            dtype[c] = "int8"

    df = pd.read_csv(path, usecols=usecols, dtype=dtype, nrows=nrows)

    # OSRCT 需要二元 treatment 和二元 outcome。
    df = df[df[treatment_col].isin([0, 1]) & df[target_col].isin([0, 1])].copy()
    df[treatment_col] = df[treatment_col].astype("int8")
    df[target_col] = df[target_col].astype("int8")

    if sample_n is not None and sample_n > 0 and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df, feature_cols


# -----------------------------
# Step 1：训练 outcome-based biasing score
# -----------------------------

def fit_outcome_bias_score(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int = 42,
) -> Tuple[np.ndarray, Pipeline]:
    """
    用 Y 训练 outcome model，得到每个样本的 outcome-related score。

    这里使用 decision_function 而不是 predict_proba，随后会标准化并重新校准均值。
    对 conversion 这种极低基准率 outcome，这样比直接使用 raw probability 更稳定。
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int8)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X, y)
    score = model.decision_function(X).astype(np.float64)
    return score, model


def make_bias_probability(
    score: np.ndarray,
    alpha: float,
    target_mean: float = 0.5,
    clip_low: float = 0.02,
    clip_high: float = 0.98,
) -> np.ndarray:
    """
    构造 P(T_s=1 | C^b)。

    score: outcome-based score，高 score 表示更可能转化。
    alpha: 混杂强度；alpha=0 时 p 为常数 target_mean，近似无混杂。
    target_mean: E[P(T_s=1|C^b)]，默认 0.5。
    clip: positivity 约束，避免概率太接近 0 或 1。

    形式：
        z_i = standardize(score_i)
        p_i = sigmoid(beta0 + alpha * z_i)
    beta0 用二分法求，使 mean(p_i) = target_mean。
    """
    score = np.asarray(score, dtype=np.float64)
    z = (score - score.mean()) / (score.std() + 1e-12)

    if not (0.0 < target_mean < 1.0):
        raise ValueError("target_mean 必须在 (0, 1) 内。")
    if not (0.0 < clip_low < clip_high < 1.0):
        raise ValueError("clip_low/clip_high 必须满足 0 < low < high < 1。")

    target_mean = float(np.clip(target_mean, clip_low + 1e-8, clip_high - 1e-8))

    if abs(alpha) < 1e-15:
        p = np.full_like(z, fill_value=target_mean, dtype=np.float64)
        return np.clip(p, clip_low, clip_high)

    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        m = sigmoid_stable(mid + alpha * z).mean()
        if m < target_mean:
            lo = mid
        else:
            hi = mid

    beta0 = 0.5 * (lo + hi)
    p = sigmoid_stable(beta0 + alpha * z)
    return np.clip(p, clip_low, clip_high)


def selected_prob_for_target_final_propensity(
    target_pi: np.ndarray,
    original_treatment_rate: float,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    """
    可选：把“希望 accepted sample 中的最终 P(T=1|X)=pi_i”
    转换成 Algorithm 2 中需要采样的 P(T_s=1|X)=r_i。

    对原始 RCT treatment rate q = P(T=1)，Algorithm 2 接受样本后：
        P(T=1 | accepted, X)
        = q r_i / [q r_i + (1-q)(1-r_i)]

    若希望它等于 pi_i，则：
        r_i = pi_i(1-q) / [q(1-pi_i) + pi_i(1-q)]

    当 q=0.5 时，r_i=pi_i；当 Criteo q≈0.85 时，二者差异很大。
    """
    q = float(original_treatment_rate)
    pi = np.asarray(target_pi, dtype=np.float64)

    numerator = pi * (1.0 - q)
    denominator = q * (1.0 - pi) + pi * (1.0 - q)
    r = numerator / (denominator + 1e-12)
    return np.clip(r, clip_low, clip_high)


# -----------------------------
# Step 2：OSRCT Algorithm 2
# -----------------------------

def osrct_sampling(
    df: pd.DataFrame,
    p_ts1: np.ndarray,
    alpha: float,
    seed: int,
    treatment_col: str = "treatment",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    精确实现 OSRCT Algorithm 2：

    对每个样本 i:
      p_i = P(T_s=1 | C_i^b)
      T_s ~ Bernoulli(p_i)
      若 T_s == T_i，保留进 D_OSRCT
      否则进入 complementary sample

    同时计算：
      p_accept_i = P(T_s = T_i | C_i^b)
                 = p_i       if T_i=1
                 = 1-p_i     if T_i=0

    complementary sample 的 Theorem 3 权重：
      w_i = p_accept_i / (1 - p_accept_i)
    """
    rng = np.random.default_rng(seed)
    p_ts1 = np.asarray(p_ts1, dtype=np.float64)

    if len(p_ts1) != len(df):
        raise ValueError("p_ts1 长度必须等于 df 行数。")

    t_obs = df[treatment_col].to_numpy(dtype=np.int8)
    t_selected = rng.binomial(1, p_ts1).astype(np.int8)

    accepted_mask = t_selected == t_obs
    rejected_mask = ~accepted_mask

    p_accept = np.where(t_obs == 1, p_ts1, 1.0 - p_ts1)
    p_reject = 1.0 - p_accept

    d_obs = df.loc[accepted_mask].copy()
    d_comp = df.loc[rejected_mask].copy()

    # 这些列是构造过程的元数据，不要给 causal estimator 当 covariates。
    d_obs["__sample_role"] = "accepted_train"
    d_obs["__osrct_alpha"] = alpha
    d_obs["__p_ts1"] = p_ts1[accepted_mask]
    d_obs["__p_accept"] = p_accept[accepted_mask]
    d_obs["__p_reject"] = p_reject[accepted_mask]
    d_obs["__complement_weight"] = np.nan

    d_comp["__sample_role"] = "complement_eval"
    d_comp["__osrct_alpha"] = alpha
    d_comp["__p_ts1"] = p_ts1[rejected_mask]
    d_comp["__p_accept"] = p_accept[rejected_mask]
    d_comp["__p_reject"] = p_reject[rejected_mask]
    d_comp["__complement_weight"] = (
        p_accept[rejected_mask] / (p_reject[rejected_mask] + 1e-12)
    )

    return d_obs, d_comp


# -----------------------------
# Step 3：ATE 与诊断
# -----------------------------

def difference_in_means(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
) -> float:
    """二元 outcome 下就是 risk difference。"""
    y1 = df.loc[df[treatment_col] == 1, outcome_col]
    y0 = df.loc[df[treatment_col] == 0, outcome_col]
    if len(y1) == 0 or len(y0) == 0:
        return float("nan")
    return float(y1.mean() - y0.mean())


def standardized_mean_difference(
    df: pd.DataFrame,
    col: str,
    treatment_col: str,
) -> float:
    """单列的 treatment-control 标准化均值差。"""
    t = df[treatment_col].to_numpy() == 1
    if t.sum() == 0 or (~t).sum() == 0:
        return float("nan")

    x = df[col].to_numpy(dtype=np.float64)
    x1, x0 = x[t], x[~t]
    denom = np.sqrt(0.5 * (np.nanvar(x1) + np.nanvar(x0))) + 1e-12
    return float((np.nanmean(x1) - np.nanmean(x0)) / denom)


def max_abs_feature_smd(
    df: pd.DataFrame,
    feature_cols: List[str],
    treatment_col: str,
) -> float:
    smds = []
    for c in feature_cols:
        try:
            smds.append(abs(standardized_mean_difference(df, c, treatment_col)))
        except Exception:
            pass
    if not smds:
        return float("nan")
    return float(np.nanmax(smds))


def summarize_sample(
    name: str,
    df_sample: pd.DataFrame,
    n_original: int,
    true_ate: float,
    feature_cols: List[str],
    outcome_col: str,
    treatment_col: str,
    alpha: float,
) -> Dict[str, float | int | str]:
    naive_ate = difference_in_means(df_sample, outcome_col, treatment_col)

    out: Dict[str, float | int | str] = {
        "sample": name,
        "alpha": float(alpha),
        "n": int(len(df_sample)),
        "accept_or_reject_rate": float(len(df_sample) / n_original),
        "treatment_rate": float(df_sample[treatment_col].mean()) if len(df_sample) else np.nan,
        "outcome_rate": float(df_sample[outcome_col].mean()) if len(df_sample) else np.nan,
        "true_ate_from_original_rct": float(true_ate),
        "naive_ate_in_this_sample": float(naive_ate),
        "naive_bias": float(naive_ate - true_ate) if np.isfinite(naive_ate) else np.nan,
        "max_abs_feature_smd": max_abs_feature_smd(df_sample, feature_cols, treatment_col)
        if len(df_sample)
        else np.nan,
    }

    if "__outcome_score" in df_sample.columns and len(df_sample):
        out["outcome_score_smd"] = standardized_mean_difference(
            df_sample, "__outcome_score", treatment_col
        )
    else:
        out["outcome_score_smd"] = np.nan

    if "__p_accept" in df_sample.columns and len(df_sample):
        out["mean_p_accept"] = float(df_sample["__p_accept"].mean())
    else:
        out["mean_p_accept"] = np.nan

    return out


def evaluate_on_complement(
    d_comp: pd.DataFrame,
    y_pred: np.ndarray,
    outcome_col: str = "conversion",
    metric: str = "mean_error",
) -> float:
    """
    用 complementary sample 做加权评估。

    y_pred 应该是模型对 d_comp 中每行 observed treatment 下 outcome 的预测。
    权重必须是：
        w_i = P(T_s = T_i | C_i^b) / [1 - P(T_s = T_i | C_i^b)]
    也就是代码中保存的 __complement_weight。

    metric:
      - mean_error: weighted average of y_pred - y
      - mae: weighted MAE
      - mse: weighted MSE
    """
    if "__complement_weight" not in d_comp.columns:
        raise ValueError("d_comp 中缺少 __complement_weight。")

    y = d_comp[outcome_col].to_numpy(dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = d_comp["__complement_weight"].to_numpy(dtype=np.float64)

    if len(y_pred) != len(y):
        raise ValueError("y_pred 长度必须等于 d_comp 行数。")

    err = y_pred - y

    if metric == "mean_error":
        return float(np.average(err, weights=w))
    if metric == "mae":
        return float(np.average(np.abs(err), weights=w))
    if metric == "mse":
        return float(np.average(err ** 2, weights=w))

    raise ValueError("metric 只能是 mean_error, mae, mse。")


# -----------------------------
# 主流程
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/criteo_train.csv")
    parser.add_argument("--outdir", default="./criteo_osrct")
    parser.add_argument("--target", default="conversion", choices=["conversion", "visit"])
    parser.add_argument("--treatment", default="treatment")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--bias-target-mean", type=float, default=0.5)
    parser.add_argument("--clip-low", type=float, default=0.02)
    parser.add_argument("--clip-high", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument(
        "--target-final-propensity",
        action="store_true",
        help=(
            "若开启，把 make_bias_probability 得到的 pi_i 解释为希望 accepted sample 中的 "
            "P(T=1|X)，再反推 Algorithm 2 的 P(T_s=1|X)。"
        ),
    )
    parser.add_argument(
        "--no-save-complement",
        action="store_true",
        help="只保存 accepted train，不保存 complementary sample。",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Reading Criteo data...")
    df, feature_cols = read_criteo(
        path=args.input,
        target_col=args.target,
        treatment_col=args.treatment,
        nrows=args.nrows,
        sample_n=args.sample_n,
        seed=args.seed,
    )

    print(f"N={len(df):,}")
    print(f"Features={feature_cols}")
    print(f"Treatment rate={df[args.treatment].mean():.6f}")
    print(f"{args.target} rate={df[args.target].mean():.6f}")

    # Ground truth ATE / risk difference from original randomized data.
    true_ate = difference_in_means(df, args.target, args.treatment)
    print(f"True ATE / risk difference from original RCT = {true_ate:.8f}")

    # 相关性诊断：仅作参考，最终 biasing score 使用所有特征训练的 outcome model。
    corr = df[feature_cols].corrwith(df[args.target]).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    corr.to_frame("corr_with_outcome").to_csv(
        os.path.join(args.outdir, f"feature_corr_with_{args.target}.csv")
    )
    print("Top feature correlations with outcome:")
    print(corr.head(10))

    print("Fitting outcome-based biasing model...")
    outcome_score, _ = fit_outcome_bias_score(
        df=df,
        feature_cols=feature_cols,
        target_col=args.target,
        seed=args.seed,
    )
    df["__outcome_score"] = outcome_score.astype("float32")

    q = float(df[args.treatment].mean())
    summary_rows = []

    for alpha in args.alphas:
        print("\n" + "=" * 80)
        print(f"Constructing OSRCT data for alpha={alpha}")

        # pi_or_p 是 outcome-related 概率曲线。
        # 默认模式：pi_or_p 直接作为 P(T_s=1|X)。
        # 可选模式：pi_or_p 作为希望 accepted sample 中的最终 P(T=1|X)，再反推 P(T_s=1|X)。
        pi_or_p = make_bias_probability(
            score=outcome_score,
            alpha=alpha,
            target_mean=args.bias_target_mean,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
        )

        if args.target_final_propensity:
            p_ts1 = selected_prob_for_target_final_propensity(
                target_pi=pi_or_p,
                original_treatment_rate=q,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
            )
        else:
            p_ts1 = pi_or_p

        d_obs, d_comp = osrct_sampling(
            df=df,
            p_ts1=p_ts1,
            alpha=alpha,
            seed=args.seed + int(round(alpha * 10000)),
            treatment_col=args.treatment,
        )

        obs_summary = summarize_sample(
            name="accepted_train",
            df_sample=d_obs,
            n_original=len(df),
            true_ate=true_ate,
            feature_cols=feature_cols,
            outcome_col=args.target,
            treatment_col=args.treatment,
            alpha=alpha,
        )
        comp_summary = summarize_sample(
            name="complement_eval",
            df_sample=d_comp,
            n_original=len(df),
            true_ate=true_ate,
            feature_cols=feature_cols,
            outcome_col=args.target,
            treatment_col=args.treatment,
            alpha=alpha,
        )

        summary_rows.extend([obs_summary, comp_summary])

        print(json.dumps(obs_summary, ensure_ascii=False, indent=2))
        print(json.dumps(comp_summary, ensure_ascii=False, indent=2))

        alpha_name = safe_alpha_name(alpha)
        mode = "finalprop" if args.target_final_propensity else "direct"

        train_path = os.path.join(
            args.outdir,
            f"criteo_osrct_{args.target}_{mode}_alpha_{alpha_name}_train.csv.gz",
        )
        d_obs.to_csv(train_path, index=False, compression="gzip")
        print(f"Saved train: {train_path}")

        if not args.no_save_complement:
            comp_path = os.path.join(
                args.outdir,
                f"criteo_osrct_{args.target}_{mode}_alpha_{alpha_name}_complement.csv.gz",
            )
            d_comp.to_csv(comp_path, index=False, compression="gzip")
            print(f"Saved complement: {comp_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, f"osrct_summary_{args.target}.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nSummary saved:", summary_path)

    metadata = {
        "input": args.input,
        "target": args.target,
        "treatment": args.treatment,
        "feature_cols": feature_cols,
        "n": int(len(df)),
        "original_treatment_rate": q,
        "original_outcome_rate": float(df[args.target].mean()),
        "true_ate_or_risk_difference": true_ate,
        "alphas": args.alphas,
        "bias_target_mean": args.bias_target_mean,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "target_final_propensity": bool(args.target_final_propensity),
        "seed": args.seed,
        "note": (
            "Columns beginning with __ are OSRCT construction metadata. "
            "Do not use them as covariates in downstream causal estimators."
        ),
    }
    meta_path = os.path.join(args.outdir, f"osrct_metadata_{args.target}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("Metadata saved:", meta_path)


if __name__ == "__main__":
    main()