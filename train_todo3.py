from __future__ import print_function, absolute_import, division

import os
import argparse
import numpy as np
import tensorflow as tf

from data_utils import *

from ecom_dfcl_pred import EcomDFCL_v3

# from base_NNmodel import basemodel
# from base_DFCL import base_DFCL

# TODO 3: 引入 test.py 中的残差分析工具
from test import analyze_residual_distribution

# -------------------- 环境设置 --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ==================== 自定义回调 ====================
class EpochMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(EpochMetricsCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "custom_epoch_metrics"))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for metric, value in logs.items():
                tf.summary.scalar(f"epoch_{metric}", value, step=epoch)
        print(f"\nEpoch {epoch + 1} metrics: {logs}")

# ==================== 主函数 ====================
def main():
    # --- 1. 配置字典 (默认值) ---
    config = {
        "model_class_name": "EcomDFCL_v3", # 默认为支持 TODO 3 的模型
        "model_path": "./model/exp_todo3_default",
        "last_model_path": "",
        "train_data": "data/criteo_train.csv",
        "val_data": "data/criteo_val.csv",
        "batch_size": 256,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "summary_steps": 1000,
        "first_decay_steps": 1000,
        "alpha": 1.2,
        "seed": 0,
        
        # TODO 3: 分布假设相关参数
        "pred_loss_dist": "bernoulli", 
        "t_df": 3.0,
        
        # 残差分析配置
        "do_residual_analysis": True,
        "residual_batches": 200,
    }

    # --- 1b. 使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Train DFCL model with configurable prediction loss distribution (TODO 3).")
    
    # 基础参数
    parser.add_argument("--model_class_name", type=str, default=config["model_class_name"])
    parser.add_argument("--model_path", type=str, default=config["model_path"])
    parser.add_argument("--last_model_path", type=str, default=config["last_model_path"])
    parser.add_argument("--train_data", type=str, default=config["train_data"])
    parser.add_argument("--val_data", type=str, default=config["val_data"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"])
    parser.add_argument("--alpha", type=float, default=config["alpha"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    
    # TODO 3 相关参数
    parser.add_argument("--pred_loss_dist", type=str, default=config["pred_loss_dist"],
                        choices=["bernoulli", "normal", "laplace", "t"],
                        help="Distribution assumption for prediction loss.")
    parser.add_argument("--t_df", type=float, default=config["t_df"], help="Degrees of freedom for Student-t distribution")
    parser.add_argument("--do_residual_analysis", action="store_true", default=config["do_residual_analysis"])
    parser.add_argument("--residual_batches", type=int, default=config["residual_batches"])

    args = parser.parse_args()

    # 更新 config
    for k in vars(args):
        config[k] = getattr(args, k)

    # 动态生成 model_path 以区分实验 (如果用户没有指定特殊的路径)
    if config['model_path'] == './model/exp_todo3_default':
        exp_name = f"{config['model_class_name']}_dist={config['pred_loss_dist']}_seed={config['seed']}"
        config["model_path"] = os.path.join("./model", exp_name)

    print("--- 运行配置 ---")
    print(f"Model Class: {config['model_class_name']}")
    print(f"Pred Loss Dist: {config['pred_loss_dist']}")
    print(f"Model Path: {config['model_path']}")
    print("--------------------")

    os.makedirs(config["model_path"], exist_ok=True)
    set_global_seed(config["seed"])

    # -------------------- 分布式策略 --------------------
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"策略中的副本数: {strategy.num_replicas_in_sync}")
    global_batch_size = config["batch_size"] * strategy.num_replicas_in_sync

    # -------------------- 数据准备 --------------------
    dataset = CSVData()

    # 训练集
    train_samples = dataset.prepare_dataset(
        config['train_data'],
        phase='train',
        batch_size=global_batch_size,
        shuffle=True
    )

    # 验证集
    val_samples = dataset.prepare_dataset(
        config['val_data'],
        phase='test',
        batch_size=global_batch_size,
        shuffle=False
    )

    # 数据转换逻辑
    label_name_list = ['treatment', 'paid', 'cost']
    drop_list = ['paid', 'cost']

    def _to_features_labels(parsed_example):
        features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}
        labels = {}
        for name in label_name_list:
            labels[name] = parsed_example[name]
        return features, labels

    train_samples = train_samples.map(_to_features_labels, num_parallel_calls=4).prefetch(1)
    val_samples = val_samples.map(_to_features_labels, num_parallel_calls=4).prefetch(1)

    # -------------------- 创建模型与优化器 --------------------
    with strategy.scope():
        # 动态获取模型类
        if config["model_class_name"] in globals():
            model_class = globals()[config["model_class_name"]]
        else:
            print(f"Error: Model class {config['model_class_name']} not found in globals. Falling back to EcomDFCL_v3.")
            model_class = EcomDFCL_v3

        # 实例化模型
        # 注意：这里去掉了 boxcox 的 transform_params_path，但保留了 TODO 3 需要的分布参数
        try:
            model = model_class(
                alpha=config["alpha"],
                pred_loss_dist=config["pred_loss_dist"],
                t_df=config["t_df"]
            )
        except TypeError:
            # 如果使用的是不接受这些参数的旧模型 (如 base_DFCL)，则回退到无参数初始化
            print(f"Warning: {config['model_class_name']} does not accept dist params. Initializing without them.")
            model = model_class()

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            config["learning_rate"],
            config["first_decay_steps"],
            t_mul=2.0, m_mul=0.9, alpha=0.01,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=100)
        
        model.compile(optimizer=optimizer, loss=None)

    # -------------------- Checkpoint --------------------
    checkpoint_dir = os.path.join(config["model_path"], "checkpoints")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print(f"从最新检查点恢复: {latest_checkpoint}")
        model.load_weights(latest_checkpoint)
    elif config["last_model_path"]:
        warmup_dir = os.path.join(config["last_model_path"], "checkpoints")
        warmup_checkpoint = tf.train.latest_checkpoint(warmup_dir)
        if warmup_checkpoint:
            print(f"从上一个模型检查点热启动: {warmup_checkpoint}")
            model.load_weights(warmup_checkpoint)

    # -------------------- Callbacks --------------------
    epoch_metrics_callback = EpochMetricsCallback(log_dir=os.path.join(config["model_path"], "logs"))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.ckpt"),
            monitor="val_total_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config["model_path"], "logs"),
            update_freq=config["summary_steps"],
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_total_loss",
            patience=10,
            mode="min",
            restore_best_weights=True,
        ),
        epoch_metrics_callback,
    ]

    # -------------------- 训练 --------------------
    print("\nStarting training...")
    model.fit(
        train_samples,
        validation_data=val_samples,
        epochs=config["num_epochs"],
        steps_per_epoch=500,
        callbacks=callbacks,
    )

    # -------------------- 保存模型 --------------------
    print(f"训练完成，正在将模型保存到: {config['model_path']}")
    model.save(config["model_path"])

    # -------------------- TODO 3: 残差分布分析 --------------------
    if config["do_residual_analysis"]:
        print(f"\n[ResidualAnalysis] 开始执行后验分析...")
        out_dir = os.path.join(config["model_path"], "residual_analysis_results")
        
        analyze_residual_distribution(
            model=model,
            dataset=val_samples,
            output_dir=out_dir,
            num_batches=config["residual_batches"],
        )
        print(f"[ResidualAnalysis] 分析完成，结果见: {out_dir}")

if __name__ == "__main__":
    main()