#!/usr/bin/env python
# coding: utf-8

"""
split_csv_gz_train_val_to_csv.py

功能：
读取 .csv.gz 文件，划分成 8:1 的 train / val，
然后输出为普通 .csv 文件。

用法：
python split_csv_gz_train_val_to_csv.py \
  --input ./criteo_osrct_811_output/criteo_osrct_conversion_direct_alpha_2p0_train.csv.gz \
  --outdir ./criteo_osrct_811_output/split_alpha_2p0 \
  --train-ratio 0.8888889 \
  --seed 42 \
  --stratify-cols treatment
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_stratify_key(df: pd.DataFrame, stratify_cols):
    if not stratify_cols:
        return None

    missing = [c for c in stratify_cols if c not in df.columns]
    if missing:
        raise ValueError(f"分层列不存在: {missing}")

    if len(stratify_cols) == 1:
        return df[stratify_cols[0]]

    return df[stratify_cols].astype(str).agg("_".join, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./criteo_osrct/criteo_osrct_conversion_direct_alpha_2p0_train.csv.gz", help="输入 .csv.gz 文件路径")
    parser.add_argument("--outdir", default="./criteo_osrct", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=8 / 9, help="8:1 时为 8/9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-cols",
        nargs="*",
        default=["treatment"],
        help="默认按 treatment 分层；不想分层则传空：--stratify-cols",
    )
    parser.add_argument("--prefix", default=None, help="输出文件名前缀")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio 必须在 0 和 1 之间")

    print(f"读取文件: {input_path}")
    df = pd.read_csv(input_path, compression="infer")

    print(f"原始数据 shape: {df.shape}")

    stratify_key = build_stratify_key(df, args.stratify_cols)

    train_df, val_df = train_test_split(
        df,
        train_size=args.train_ratio,
        test_size=1.0 - args.train_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_key,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if args.prefix is not None:
        prefix = args.prefix
    else:
        name = input_path.name
        if name.endswith(".csv.gz"):
            prefix = name[:-7]
        elif name.endswith(".csv"):
            prefix = name[:-4]
        else:
            prefix = input_path.stem

    train_path = outdir / f"{prefix}_split8_train.csv"
    val_path = outdir / f"{prefix}_split1_val.csv"

    print(f"保存 train CSV: {train_path}")
    train_df.to_csv(train_path, index=False)

    print(f"保存 val CSV: {val_path}")
    val_df.to_csv(val_path, index=False)

    print("\n划分完成")
    print(f"train shape: {train_df.shape}, ratio={len(train_df) / len(df):.6f}")
    print(f"val   shape: {val_df.shape}, ratio={len(val_df) / len(df):.6f}")

    if "treatment" in df.columns:
        print("\nTreatment 分布检查:")
        check = pd.DataFrame({
            "original": df["treatment"].value_counts(normalize=True).sort_index(),
            "train": train_df["treatment"].value_counts(normalize=True).sort_index(),
            "val": val_df["treatment"].value_counts(normalize=True).sort_index(),
        })
        print(check)

    if "__complement_weight" in df.columns:
        print("\n__complement_weight 均值检查:")
        print(f"original mean weight: {df['__complement_weight'].mean():.6f}")
        print(f"train    mean weight: {train_df['__complement_weight'].mean():.6f}")
        print(f"val      mean weight: {val_df['__complement_weight'].mean():.6f}")


if __name__ == "__main__":
    main()