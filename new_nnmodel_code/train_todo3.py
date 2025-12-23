from __future__ import print_function, absolute_import, division

import os
import argparse
import numpy as np
import tensorflow as tf

from data_utils import *
from base_NNmodel import basemodel
from base_DFCL import base_DFCL
from res_base_DFCL import res_base_DFCL
from DCN_base_DFCL import DCN_base_DFCL
from SEBlock_base_DFCL import SEBlock_base_DFCL
from GLU_base_DFCL_predloss import GLU_base_DFCL, DENSE_FEATURE_NAME  # 需要在该文件中加入 TODO 3 的 pred_loss_dist 支持
from preprocess_boxcox import find_optimal_transformations_from_dataset

# 残差分析工具（TODO 3）
# - 建议把 test.py 里 ResidualExtractor 改成支持 paid/cost 的通用版本
from residual_analysis import analyze_residual_distribution

# -------------------- 环境设置 --------------------
# 禁用 GPU（如果你需要 GPU，把这一行删掉）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 只显示 Error，隐藏大多数 TF INFO/WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 若想更稳定复现，可打开确定性算子（可能会降低速度）
# os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ==================== 自定义回调：按 epoch 打印 metrics ====================
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


def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    # -------------------- 默认配置 --------------------
    config = {
        "model_class_name": "GLU_base_DFCL",
        "model_path": "./model/GLU_base_DFCL_2pll_alpha=1.2_predloss=bernoulli_seed=0",
        "loss_function": "2pll",
        "alpha": 1.2,
        "pred_loss_dist": "bernoulli",  # TODO 3: bernoulli/normal/laplace/t/beta/gamma
        "t_df": 3.0,                   # Student-t df
        "seed": 0,

        "last_model_path": "",
        "train_data": "data/criteo_train.csv",
        "val_data": "data/criteo_val.csv",
        "batch_size": 256,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "summary_steps": 1000,
        "first_decay_steps": 1000,

        # residual analysis
        "do_residual_analysis": True,
        "residual_batches": 200,
    }

    # -------------------- CLI 参数 --------------------
    parser = argparse.ArgumentParser(description="Train DFCL model with configurable prediction loss distribution (TODO 3).")
    parser.add_argument("--model_class_name", type=str, default=config["model_class_name"])
    parser.add_argument("--model_path", type=str, default=config["model_path"])
    parser.add_argument("--loss_function", type=str, default=config["loss_function"])
    parser.add_argument("--alpha", type=float, default=config["alpha"])

    parser.add_argument("--pred_loss_dist", type=str, default=config["pred_loss_dist"],
                        choices=["bernoulli", "normal", "laplace", "t", "beta", "gamma"],
                        help="Residual distribution assumption used in prediction loss (TODO 3).")
    parser.add_argument("--t_df", type=float, default=config["t_df"], help="df for Student-t loss (only when pred_loss_dist=t).")
    parser.add_argument("--seed", type=int, default=config["seed"])

    parser.add_argument("--train_data", type=str, default=config["train_data"])
    parser.add_argument("--val_data", type=str, default=config["val_data"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"])
    parser.add_argument("--summary_steps", type=int, default=config["summary_steps"])
    parser.add_argument("--first_decay_steps", type=int, default=config["first_decay_steps"])
    parser.add_argument("--last_model_path", type=str, default=config["last_model_path"])

    parser.add_argument("--do_residual_analysis", action="store_true", default=config["do_residual_analysis"])
    parser.add_argument("--residual_batches", type=int, default=config["residual_batches"])

    args = parser.parse_args()

    # -------------------- 应用 CLI 到 config --------------------
    for k in vars(args):
        config[k] = getattr(args, k)

    print("--- 运行配置 ---")
    print(f"Model Class: {config['model_class_name']}")
    print(f"Model Path: {config['model_path']}")
    print(f"Decision Loss Function: {config['loss_function']}")
    print(f"Alpha: {config['alpha']}")
    print(f"Pred Loss Dist: {config['pred_loss_dist']}")
    print(f"Seed: {config['seed']}")
    print("--------------------")

    os.makedirs(config["model_path"], exist_ok=True)
    set_global_seed(config["seed"])

    # -------------------- 分布式策略 --------------------
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"策略中的副本数: {strategy.num_replicas_in_sync}")
    global_batch_size = config["batch_size"] * strategy.num_replicas_in_sync

    # -------------------- 数据 --------------------
    dataset = CSVData()

    raw_train_samples = dataset.prepare_dataset(
        config['train_data'],
        phase='train',
        batch_size=global_batch_size,
        shuffle=True
    )

    raw_val_samples = dataset.prepare_dataset(
        config['val_data'],
        phase='test',
        batch_size=global_batch_size,
        shuffle=False
    )

    # 你的 label/drop 逻辑保持不变
    label_name_list = ['treatment', 'paid', 'cost']
    drop_list = ['paid', 'cost']

    def _to_features_labels(parsed_example):
        # features：除了 drop_list 的都保留
        features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}

        labels = {}
        for name in label_name_list:
            labels[name] = parsed_example[name]

        return features, labels

    # ====================  预处理：从 raw_train_samples 抽样生成 transform 参数 ====================
    # ✅ 一定要用 join，避免你日志里那种 "\dense_transform..." 的路径拼接问题
    os.makedirs(config["model_path"], exist_ok=True)
    config["transform_params_path"] = os.path.join(config["model_path"], "dense_transform_params.json")

    if not tf.io.gfile.exists(config["transform_params_path"]):
        print(f"[Preprocess] generating transform params -> {config['transform_params_path']}")
        # 注意：这里直接用 raw_train_samples（每个元素是 parsed_example dict），不需要 TFRecord
        find_optimal_transformations_from_dataset(
            ds=raw_train_samples,
            dense_feature_names=DENSE_FEATURE_NAME,
            output_path=config["transform_params_path"],
            max_samples=100000
        )

    print(f"[Preprocess] transform params ready: {config['transform_params_path']}")

    # ==================== 6. dataset 转成训练格式 ====================
    train_samples = raw_train_samples.map(
        _to_features_labels,
        num_parallel_calls=4   # 你原来写的“安全值”保留
    ).prefetch(1)

    val_samples = raw_val_samples.map(
        _to_features_labels,
        num_parallel_calls=4
    ).prefetch(1)


    # -------------------- 创建模型与优化器 --------------------
    with strategy.scope():
        model_class = globals()[config["model_class_name"]]

        # NOTE:
        # 需要在 GLU_base_DFCL_fcd_2.py 中的 __init__ 增加：
        #   pred_loss_dist / t_df 的解析，并在 compute_local_losses 中按分布切换 loss
        model = model_class(
            alpha=config["alpha"],
            loss_function=config["loss_function"],
            pred_loss_dist=config["pred_loss_dist"],
            t_df=config["t_df"],
            transform_params_path=config["transform_params_path"],
        )

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            config["learning_rate"],
            config["first_decay_steps"],
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.01,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=100)

        model.compile(optimizer=optimizer, loss=None)

    # -------------------- checkpoint --------------------
    checkpoint_dir = os.path.join(config["model_path"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
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

    # -------------------- callbacks --------------------
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

    # -------------------- 残差分布分析（TODO 3 的后验检验） --------------------
    if config["do_residual_analysis"]:
        out_dir = os.path.join(
            config["model_path"],
            f"residual_analysis_predloss={config['pred_loss_dist']}_seed={config['seed']}",
        )
        print(f"\n[ResidualAnalysis] output_dir = {out_dir}")
        analyze_residual_distribution(
            model=model,
            dataset=val_samples,
            output_dir=out_dir,
            num_batches=config["residual_batches"],
        )


if __name__ == "__main__":
    main()
