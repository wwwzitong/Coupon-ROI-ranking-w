from __future__ import print_function, absolute_import, division

import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

from data_utils import *

from ecom_dfcl_SDP import EcomDFCL_v3, RobustnessSDP

# from base_NNmodel import basemodel
# from base_DFCL import base_DFCL

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


class SDPVerificationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, log_dir, check_freq=1, epsilon=0.1, output_file=None):
        """
        Args:
            val_dataset: 用于验证的数据集 (tf.data.Dataset)
            log_dir: TensorBoard 日志目录
            check_freq: 每多少个 epoch 检查一次
            epsilon: 鲁棒性校验的扰动半径阈值
        """
        super(SDPVerificationCallback, self).__init__()
        # 从 dataset 中预取一个 batch 用于固定验证
        # 注意：这里假设 val_dataset 已经经过 batch 处理
        try:
            iterator = iter(val_dataset)
            self.sample_batch = next(iterator)
            self.features, _ = self.sample_batch # 解包 (features, labels)
            print("SDP Callback: 成功提取验证样本 batch。")
        except Exception as e:
            print(f"SDP Callback Warning: 无法从验证集提取样本，SDP 验证将被跳过。错误: {e}")
            self.features = None

        # 创建独立的 writer 以免混淆
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, "sdp_robustness"))
        self.check_freq = check_freq
        self.epsilon = epsilon
        self.output_file = output_file or os.path.join(log_dir, "sdp_metrics.txt")

    def on_epoch_end(self, epoch, logs=None):
        if self.features is None or (epoch + 1) % self.check_freq != 0:
            return
            
        print(f"\n[SDP Callback] 正在进行第 {epoch+1} 轮的鲁棒性验证 (Epsilon={self.epsilon})...")
        
        # 调用 RobustnessSDP 静态方法进行验证
        # 注意：RobustnessSDP 必须在 ecom_dfcl_SDP.py 中定义
        try:
            metrics = RobustnessSDP.verify_decision_robustness(
                self.model, 
                self.features, 
                epsilon=self.epsilon
            )
            
            # 打印简报
            print(f"  > Lipschitz Constant (L): {metrics['lipschitz_constant']:.4f}")
            print(f"  > Avg_Decision_Margin: {metrics['avg_margin']:.4f}")
            print(f"  > Avg Safe Radius: {metrics['avg_safe_radius']:.4f}")
            print(f"  > Robust Samples Ratio: {metrics['robustness_ratio']:.2%}")
            
            # 追加写入文件
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_text = (
                f"[{ts}] epoch={epoch+1} epsilon={self.epsilon}\n"
                f"  > Lipschitz Constant (L): {metrics['lipschitz_constant']:.4f}\n"
                f"  > Avg_Decision_Margin: {metrics['avg_margin']:.4f}\n"
                f"  > Avg Safe Radius: {metrics['avg_safe_radius']:.4f}\n"
                f"  > Robust Samples Ratio: {metrics['robustness_ratio']:.2%}\n\n"
            )
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log_text)

            # 记录到 TensorBoard
            with self.file_writer.as_default():
                tf.summary.scalar("SDP/Lipschitz_Constant_L", metrics['lipschitz_constant'], step=epoch)
                tf.summary.scalar("SDP/Avg_Decision_Margin", metrics['avg_margin'], step=epoch) # 注意键名匹配
                tf.summary.scalar("SDP/Avg_Safe_Radius", metrics['avg_safe_radius'], step=epoch)
                tf.summary.scalar("SDP/Robustness_Ratio", metrics['robustness_ratio'], step=epoch)
                
        except Exception as e:
            print(f"\n[SDP Callback Error] 验证失败: {e}")

# ==================== 主函数 ====================
def main():
    # --- 1. 配置字典 (默认值) ---
    config = {
        "model_class_name": "EcomDFCL_v3", 
        "model_path": "./model/exp_todo3_default",
        "last_model_path": "",
        "train_data": "data/criteo_train.csv",
        "val_data": "data/criteo_val.csv",
        "batch_size": 256,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "summary_steps": 1000,
        "first_decay_steps": 1000,
        "alpha": 0.1,
        "seed": 0,

        # 新增配置
        "sdp_check_freq": 1,   # 每几个 epoch 检查一次
        "sdp_epsilon": 0.1,    # 鲁棒性验证的扰动半径
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
    # 新增参数
    parser.add_argument("--sdp_check_freq", type=int, default=config["sdp_check_freq"], help="Frequency of SDP checks (epochs)")
    parser.add_argument("--sdp_epsilon", type=float, default=config["sdp_epsilon"], help="Epsilon for robustness verification")

    args = parser.parse_args()

    # 更新 config
    for k in vars(args):
        config[k] = getattr(args, k)

    print("--- 运行配置 ---")
    print(f"Model Class: {config['model_class_name']}")
    print(f"Alpha: {config['alpha']}")
    print(f"Model Path: {config['model_path']}")
    print("--------------------")

    os.makedirs(config["model_path"], exist_ok=True)

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
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5e3)
        
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
    # 新增：实例化 SDP 回调
    sdp_callback = SDPVerificationCallback(
        val_dataset=val_samples,
        log_dir=os.path.join(config["model_path"], "logs"),
        check_freq=config["sdp_check_freq"],
        epsilon=config["sdp_epsilon"]
    )
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
        sdp_callback,
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

if __name__ == "__main__":
    main()