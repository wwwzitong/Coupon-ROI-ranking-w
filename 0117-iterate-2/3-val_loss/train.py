from __future__ import print_function, absolute_import, division
import os
import sys
import tensorflow as tf
import numpy as np
import math
import json
import shutil
import argparse
import random
import io
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from dfcl_regretNet_v1_rplusc import EcomDFCL_regretNet_rplusc, DENSE_FEATURE_NAME
# from dfcl_regretNet_v2_tau import EcomDFCL_regretNet_tau, DENSE_FEATURE_NAME

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_utils import *

import argparse # 导入 argparse 模块
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有 GPU，自然不会加载 CUDA。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息（隐藏 INFO 和 WARNING）

# ==================== 设置随机种子确保可复现性 ====================
def set_seeds(seed=42):
    """
    设置所有随机种子以确保实验可复现
    Args:
        seed: 随机种子值，默认为42
    """
    # 设置Python随机种子
    random.seed(seed)
    
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置TensorFlow随机种子
    tf.random.set_seed(seed)
    # 设置操作确定性（可能影响性能但提高可复现性）
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # 设置PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"已设置随机种子: {seed}")

set_seeds(42)  # 你可以更改为任何固定值

# 强制UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# In[7]:


# ==================== 建议添加的回调 ====================
class EpochMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(EpochMetricsCallback, self).__init__()
        # You can now use the log_dir, for example, to create a summary writer
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "custom_epoch_metrics"))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Example of how you might use the writer to log metrics at the end of an epoch
        with self.writer.as_default():
            for metric, value in logs.items():
                tf.summary.scalar(f"epoch_{metric}", value, step=epoch)
        
        # Your existing on_epoch_end logic can go here
        print(f"\nEpoch {epoch+1} metrics: {logs}")


# In[8]:


# --- 1. 配置字典（替代命令行参数） ---
config = {
    'model_class_name': 'EcomDFCL_regretNet_rc',
    'model_path': './model/EcomDFCL_regretNet_rc_2pll_2pos_lr3',
    'last_model_path': '',
    'train_data': '../../data/criteo_train.csv', 
    'val_data': '../../data/criteo_val.csv',
    'batch_size': 256,
    'num_epochs': 50,
    'learning_rate': 0.001, # initial learning rate
    'summary_steps': 1000,
    'first_decay_steps': 1000,
    'clipnorm': 5e3,
    'max_multiplier': 1.0,
    'scheduler': 'raw',
    'tau': 1.0,
    'rho': 0.1,
}

# --- 1b. 使用 argparse 解析命令行参数 ---
parser = argparse.ArgumentParser(description='Train a model for Criteo dataset.')
parser.add_argument('--model_class_name', type=str, default=config['model_class_name'],
                    help='The name of the model class to train.')
parser.add_argument('--model_path', type=str, default=config['model_path'],
                    help='The path to save the model and logs.')
parser.add_argument('--fcd_mode', type=str, default="log1p", help='Fcd mode: raw or log1p.')
parser.add_argument('--clipnorm', type=float, default=5e3, help='Gradient clipnorm')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--max_multiplier', type=float, default=1.0, help='max lagrangian multiplier')
parser.add_argument('--scheduler', type=str, default='raw', help='learning rate scheduler')
parser.add_argument('--tau', type=float, default=1.0, help='temprature tau')
parser.add_argument('--rho', type=float, default=0.1, help='rho for updating mu')


args = parser.parse_args()

# 使用命令行参数更新 config 字典
config['model_class_name'] = args.model_class_name
config['model_path'] = args.model_path
config['fcd_mode'] = args.fcd_mode
config['clipnorm'] = args.clipnorm
config['learning_rate'] = args.lr
config['tau'] = args.tau
config['rho'] = args.rho
config['max_multiplier'] = args.max_multiplier
config['scheduler'] = args.scheduler

print("--- 运行配置 ---")
print(f"Model Class: {config['model_class_name']}")
print(f"Model Path: {config['model_path']}")
print(f"tau: {config['tau']}")
print(f"rho: {config['rho']}")
print(f"FCD Mode: {config['fcd_mode']}")
print(f"clipnorm: {config['clipnorm']}")
print(f"learning rate: {config['learning_rate']}")
print(f"scheduler: {config['scheduler']}")
print(f"max_multiplier: {config['max_multiplier']}")
print("--------------------")

# In[9]:
# --- Main execution block for demonstration ---
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print(f'策略中的副本数: {strategy.num_replicas_in_sync}')
global_batch_size = config['batch_size'] * strategy.num_replicas_in_sync
# In[10]:


# ==================== 每次都从头开始：删除旧模型目录，禁止从检查点恢复 ====================
if os.path.exists(config['model_path']):
    print(f"[Reset] 删除已有模型目录: {config['model_path']}")
    shutil.rmtree(config['model_path'])

# 重建必要目录
os.makedirs(config['model_path'], exist_ok=True)

checkpoint_dir = os.path.join(config['model_path'], 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

log_dir = os.path.join(config['model_path'], 'logs')
os.makedirs(log_dir, exist_ok=True)

print("[Reset] 已清空并重建输出目录，将从随机初始化开始训练（不恢复任何 checkpoint）。")


# 添加自定义回调
epoch_metrics_callback = EpochMetricsCallback(log_dir=os.path.join(config['model_path'], 'logs'))
callbacks = [
    # 修改 ModelCheckpoint 来只保存最佳模型
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.ckpt'), # 使用一个固定的文件名
        monitor='val_total_loss', # 监控与 EarlyStopping 相同的指标
        save_best_only=True,      # 只保存最佳模型
        save_weights_only=True,   # 只保存权重
        mode='min'                # 监控的指标越小越好
    ),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config['model_path'], 'logs'), update_freq=config['summary_steps']),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    ),
    epoch_metrics_callback,
]


# In[11]:


# 数据
dataset = CSVData()
train_samples = dataset.prepare_dataset(config['train_data'], phase='train', batch_size=global_batch_size, shuffle=True)
# --- Step: 提取 drop_list 和 label_name_list ---
label_name_list = ['treatment','paid','cost']
drop_list = ['paid','cost']
# --- Step: 将 dataset 转换为 (features, labels) 格式 ---
def _to_features_labels(parsed_example):
    # 提取 features（从 feature_name_list 中）
    features = {name: parsed_example[name] for name in parsed_example if name not in drop_list}
    # 构建 labels 字典，特别处理 _treatment_index 的反转
    labels = {}
    for name in label_name_list:
        value = parsed_example[name]
        labels[name] = value

    return features, labels  # 返回 (features, labels) 其中 labels 是 dict
# --- 应用 map 转换 ---
train_samples = train_samples.map(
    _to_features_labels,
    num_parallel_calls=4  # ✅ 安全值，不要用 AUTOTUNE
).prefetch(1)  # ✅ 避免缓冲区过大


val_samples = dataset.prepare_dataset(config['val_data'], phase='test', batch_size=global_batch_size, shuffle=False)
# --- 应用 map 转换 ---
val_samples = val_samples.map(
    _to_features_labels,
    num_parallel_calls=4  # ✅ 安全值，不要用 AUTOTUNE
).prefetch(1)  # ✅ 避免缓冲区过大

# In[ ]:

# 去掉了steps_per_epoch = 300, 
# ==================== 预计算全局统计量 ====================

# ==================== Step 1: 预计算全局统计量（均值/方差） ====================
def _count_csv_rows(csv_path: str) -> int:
    # 统计数据行数（去掉 header）
    n = 0
    with tf.io.gfile.GFile(csv_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
        n = max(i, 0)  # i 是最后一行索引；减去 header 后约等于 i
    return n

def _welford_merge(count_a, mean_a, M2_a, count_b, mean_b, M2_b):
    delta = mean_b - mean_a
    count = count_a + count_b
    mean = mean_a + delta * (count_b / tf.maximum(count, 1.0))
    M2 = M2_a + M2_b + tf.square(delta) * (count_a * count_b / tf.maximum(count, 1.0))
    return count, mean, M2

def compute_global_dense_stats(ds, dense_names, clip_min=0.0):
    d = len(dense_names)
    count = tf.constant(0.0, dtype=tf.float64)

    mean_raw = tf.zeros([d], dtype=tf.float64)
    M2_raw = tf.zeros([d], dtype=tf.float64)

    mean_log = tf.zeros([d], dtype=tf.float64)
    M2_log = tf.zeros([d], dtype=tf.float64)

    for features, _ in ds:
        x = tf.stack([tf.cast(features[name], tf.float64) for name in dense_names], axis=1)  # [B, d]
        x = tf.maximum(x, tf.cast(clip_min, tf.float64))

        # raw
        b_count = tf.cast(tf.shape(x)[0], tf.float64)
        b_mean = tf.reduce_mean(x, axis=0)
        b_M2 = tf.reduce_sum(tf.square(x - b_mean), axis=0)
        count, mean_raw, M2_raw = _welford_merge(count, mean_raw, M2_raw, b_count, b_mean, b_M2)

        # log1p(raw)
        xl = tf.math.log1p(x)
        b_mean_l = tf.reduce_mean(xl, axis=0)
        b_M2_l = tf.reduce_sum(tf.square(xl - b_mean_l), axis=0)
        _, mean_log, M2_log = _welford_merge(count - b_count, mean_log, M2_log, b_count, b_mean_l, b_M2_l)

    denom = tf.maximum(count - 1.0, 1.0)
    std_raw = tf.sqrt(tf.maximum(M2_raw / denom, 0.0))
    std_log = tf.sqrt(tf.maximum(M2_log / denom, 0.0))

    # 防止 std 为 0
    std_raw = tf.where(std_raw > 0.0, std_raw, tf.ones_like(std_raw))
    std_log = tf.where(std_log > 0.0, std_log, tf.ones_like(std_log))

    return {
        "raw":   {"mean": mean_raw.numpy().tolist(), "std": std_raw.numpy().tolist()},
        "log1p": {"mean": mean_log.numpy().tolist(), "std": std_log.numpy().tolist()},
    }

# 只走一遍训练集（防止 prepare_dataset 内部 repeat 导致无限）
num_rows = _count_csv_rows(config['train_data'])
num_batches = int(math.ceil(num_rows / float(global_batch_size)))

# # ===== 自动计算 steps_per_epoch（避免写死 500）=====
# steps_per_epoch = config.get("steps_per_epoch", None)
# if steps_per_epoch is None or steps_per_epoch <= 0:
#     steps_per_epoch = num_batches  # 训练集完整跑一遍

# # （可选）也给验证集算 validation_steps，避免验证集被截断或不完整
# val_rows = _count_csv_rows(config['val_data'])
# validation_steps = int(math.ceil(val_rows / float(global_batch_size)))


train_for_stats = dataset.prepare_dataset(
    config['train_data'], phase='train', batch_size=global_batch_size, shuffle=False
).map(_to_features_labels, num_parallel_calls=4).take(num_batches).prefetch(1)

dense_stats = compute_global_dense_stats(train_for_stats, DENSE_FEATURE_NAME, clip_min=0.0)


# import tensorflow_addons as tfa

# 在策略范围内创建模型和优化器
with strategy.scope():
    # 从配置中动态获取并实例化模型类
    model_class = globals()[config['model_class_name']]
    model = model_class(tau=config['tau'], rho=config['rho'], max_multiplier=config['max_multiplier'], fcd_mode=config['fcd_mode'], dense_stats=dense_stats)
    
    if config['scheduler'] == 'raw':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipnorm=config['clipnorm'])
        #optimizer = tfa.optimizers.AdamW(learning_rate=config['learning_rate'], weight_decay=1e-4, clipnorm=5e3)   
    elif config['scheduler'] == 'warmup+decay': 
        # 学习率调度器：带 Warmup 的余弦退火
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            config['learning_rate'],
            config['first_decay_steps'],
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.01  # 最小学习率是初始值的 1%
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config['clipnorm'])
    else: # 默认值
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipnorm=config['clipnorm'])

    model.compile(
        optimizer=optimizer,loss=None
    )


model.fit(train_samples, validation_data=val_samples, epochs=config['num_epochs'], steps_per_epoch = 500, callbacks=callbacks) # ,verbose=2) # 只在每个 epoch 结束后打印一行日志

# 保存最终模型
print(f"训练完成，正在将模型保存到: {config['model_path']}")
model.save(config['model_path'])