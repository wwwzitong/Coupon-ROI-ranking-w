#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export one batch from your existing tf.data.Dataset (CSVData pipeline)
to sample_features.npz (and optionally labels.npz).

Run:
  python export_one_batch_npz_from_csvdata.py \
    --out sample_features.npz \
    --labels_out sample_labels.npz \
    --split val

You must ensure:
- CSVData can be imported
- config dict is available (either import from your config module or load from json)
"""

import argparse
import json
import numpy as np
import tensorflow as tf
from typing import Optional

# ========= 你需要改：把 CSVData 的真实 import 路径填对 =========
import os
import sys
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
# from data_utils_ECLIFT import *
from data_utils import *



def build_datasets_from_config(config: dict, global_batch_size: int):
    """
    Exactly mirrors your training pipeline: prepare_dataset + map(_to_features_labels) + prefetch
    """
    dataset = CSVData()

    train_samples = dataset.prepare_dataset(
        config["train_data"], phase="train", batch_size=global_batch_size, shuffle=True
    )
    val_samples = dataset.prepare_dataset(
        config["val_data"], phase="test", batch_size=global_batch_size, shuffle=False
    )

    label_name_list = ["treatment", "paid", "cost"]
    drop_list = ["paid", "cost"]

    def _to_features_labels(parsed_example):
        # features: drop paid/cost
        features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}
        # labels: treatment/paid/cost
        labels = {name: parsed_example[name] for name in label_name_list}
        return features, labels

    train_samples = train_samples.map(_to_features_labels, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_samples   = val_samples.map(_to_features_labels,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_samples, val_samples


def _save_tensor_dict_to_npz(path: str, tensor_dict: dict):
    """
    Save a dict[str, Tensor] to npz.
    Supports Tensor / RaggedTensor / SparseTensor.
    """
    to_save = {}

    for k, v in tensor_dict.items():
        if tf.is_tensor(v):
            to_save[k] = v.numpy()
        elif isinstance(v, tf.RaggedTensor):
            to_save[f"{k}__type"] = np.array(["ragged"], dtype=object)
            to_save[f"{k}__values"] = v.values.numpy()
            to_save[f"{k}__row_splits"] = v.row_splits.numpy()
        elif isinstance(v, tf.SparseTensor):
            to_save[f"{k}__type"] = np.array(["sparse"], dtype=object)
            to_save[f"{k}__indices"] = v.indices.numpy()
            to_save[f"{k}__values"] = v.values.numpy()
            to_save[f"{k}__dense_shape"] = v.dense_shape.numpy()
        else:
            to_save[k] = np.asarray(v)

    np.savez(path, **to_save)
    print(f"[OK] Saved: {path}")
    print(f"[INFO] keys: {list(to_save.keys())}")


def export_one_batch(ds: tf.data.Dataset, out_path: str, labels_out_path: Optional[str] = None):
    """
    Take exactly one batch from ds which yields (features, labels)
    """
    features, labels = next(iter(ds))

    if not isinstance(features, dict):
        raise ValueError("Expected features to be dict, but got: " + str(type(features)))
    if not isinstance(labels, dict):
        raise ValueError("Expected labels to be dict, but got: " + str(type(labels)))

    _save_tensor_dict_to_npz(out_path, features)

    if labels_out_path:
        _save_tensor_dict_to_npz(labels_out_path, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True,
                        help="Train data path. Support single path or comma-separated list.")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Val/Test data path. Support single path or comma-separated list.")
    parser.add_argument("--batch_size", type=int, default=256, help="Global batch size used in dataset.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Which dataset to sample from.")
    parser.add_argument("--out", type=str, default="sample_features.npz", help="Output .npz for features.")
    parser.add_argument("--labels_out", type=str, default="sample_labels.npz",
                        help="Optional output .npz for labels (set empty to disable).")
    args = parser.parse_args()

    # Build a minimal config dict compatible with your build_datasets_from_config()
    config = {
        "train_data": args.train_data,
        "val_data": args.val_data,
    }

    train_ds, val_ds = build_datasets_from_config(config, global_batch_size=args.batch_size)
    ds = train_ds if args.split == "train" else val_ds

    labels_out = args.labels_out.strip()
    if labels_out == "":
        labels_out = None

    export_one_batch(ds, out_path=args.out, labels_out_path=labels_out)

    print("\nNext step:")
    print(f"  export SAMPLE_FEATURES_NPZ={args.out}")
    print("  python sdp_lipschitz.py")


if __name__ == "__main__":
    main()
