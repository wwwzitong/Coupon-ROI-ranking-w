from __future__ import print_function, absolute_import, division

import os
import argparse
import inspect
import tensorflow as tf
import numpy as np
import os
import sys
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from data_utils import *
from ecom_dfcl_boxcox import EcomDFCL_v3, DENSE_FEATURE_NAME
# from base_NNmodel import basemodel
# from base_DFCL import base_DFCL
# from res_base_DFCL import res_base_DFCL
# from DCN_base_DFCL import DCN_base_DFCL
# from SEBlock_base_DFCL import SEBlock_base_DFCL
# from GLU_base_DFCL_fcd_2 import GLU_base_DFCL, DENSE_FEATURE_NAME

# ✅ 改：从 dataset 抽样生成 boxcox/yj 参数
from preprocess_boxcox import find_optimal_transformations_from_dataset


# =============== 环境变量（保持你原来的设置）===============
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有 GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 只显示错误信息（隐藏 INFO 和 WARNING）


# ==================== 建议添加的回调 ====================
class EpochMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(EpochMetricsCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "custom_epoch_metrics"))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for metric, value in logs.items():
                tf.summary.scalar(f"epoch_{metric}", value, step=epoch)
        print(f"\nEpoch {epoch+1} metrics: {logs}")


# ==================== 1. 默认配置 ====================
config = {
    'model_class_name': 'EcomDFCL_v3',
    'model_path': './model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=0.1',
    'loss_function': '2pll',  # 3erl, ...
    'alpha': 0.1,
    'last_model_path': '',
    'train_data': '../../data/criteo_train.csv',
    'val_data': '../../data/criteo_val.csv',
    'batch_size': 256,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'summary_steps': 1000,
    'first_decay_steps': 1000,
    'transform_params_path': '',
    'clipnorm': 5e3,
}


# ==================== 2. argparse 覆盖配置 ====================
parser = argparse.ArgumentParser(description='Train a model for Criteo dataset.')
parser.add_argument('--model_class_name', type=str, default=config['model_class_name'],
                    help='The name of the model class to train.')
parser.add_argument('--model_path', type=str, default=config['model_path'],
                    help='The path to save the model and logs.')
parser.add_argument('--loss_function', type=str, default=config['loss_function'],
                    help='The expression of decision loss function.')
parser.add_argument('--alpha', type=float, default=config['alpha'],
                    help='Alpha value for the loss function.')
parser.add_argument('--clipnorm', type=float, default=5e3, help='Gradient clipnorm')

args = parser.parse_args()

config['model_class_name'] = args.model_class_name
config['model_path'] = args.model_path
config['loss_function'] = args.loss_function
config['alpha'] = args.alpha
config['clipnorm'] = args.clipnorm


print("--- 运行配置 ---")
print(f"Model Class: {config['model_class_name']}")
print(f"Model Path: {config['model_path']}")
print(f"Decision Loss Function: {config['loss_function']}")
print(f"Alpha: {config['alpha']}")
print(f"clipnorm: {config['clipnorm']}")
print("--------------------")


# ==================== 3. 分布式策略 & batch ====================
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print(f'策略中的副本数: {strategy.num_replicas_in_sync}')
global_batch_size = config['batch_size'] * strategy.num_replicas_in_sync


# ==================== 4. 构建原始 dataset（先于模型创建）====================
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


# ==================== 5. ✅ 预处理：从 raw_train_samples 抽样生成 transform 参数 ====================
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


# ==================== 7. 创建模型（把 transform_params_path 传给支持它的模型类）====================
with strategy.scope():
    model_class = globals()[config['model_class_name']]

    init_kwargs = dict(alpha=config['alpha'], loss_function=config['loss_function'])

    # ✅ 只有 GLU_base_DFCL 这种构造函数支持 transform_params_path 才传，避免 base_DFCL 报参数不匹配
    if "transform_params_path" in inspect.signature(model_class.__init__).parameters:
        init_kwargs["transform_params_path"] = config["transform_params_path"]

    model = model_class(**init_kwargs)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        config['learning_rate'],
        config['first_decay_steps'],
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5e3)

    model.compile(optimizer=optimizer, loss=None)


# ==================== 8. 加载检查点 ====================
checkpoint_dir = os.path.join(config['model_path'], 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    print(f"从最新检查点恢复: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

elif config['last_model_path']:
    warmup_dir = os.path.join(config['last_model_path'], 'checkpoints')
    warmup_checkpoint = tf.train.latest_checkpoint(warmup_dir)
    if warmup_checkpoint:
        print(f"从上一个模型检查点热启动: {warmup_checkpoint}")
        model.load_weights(warmup_checkpoint)


# ==================== 9. callbacks ====================
log_dir = os.path.join(config['model_path'], 'logs')
os.makedirs(log_dir, exist_ok=True)

epoch_metrics_callback = EpochMetricsCallback(log_dir=log_dir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.ckpt'),
        monitor='val_total_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    ),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=config['summary_steps']),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    ),
    epoch_metrics_callback,
]


# ==================== 10. 训练 & 保存 ====================
model.fit(
    train_samples,
    validation_data=val_samples,
    epochs=config['num_epochs'],
    steps_per_epoch=500,   # 你原来写的保持不变
    callbacks=callbacks
)

print(f"训练完成，正在将模型保存到: {config['model_path']}")
model.save(config['model_path'])
