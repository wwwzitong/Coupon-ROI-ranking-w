from __future__ import print_function, absolute_import, division
import os
import tensorflow as tf
import argparse
import random
# from tensorflow import keras
import numpy as np
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from ecom_slearner import SLearner

import os
import sys
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_utils import *

# from ecom_dfcl_copy_0926 import EcomDFCL_re
# from ecom_drm import EcomDRM19
# from ecom_dfl import EcomDFL
# from ecom_dfcl_gradnorm import EcomDFCL_gradnorm
# from ecom_slearner import EcomDFCL_only_pl
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有 GPU，自然不会加载 CUDA。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息（隐藏 INFO 和 WARNING）

import io

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
    'model_class_name': 'SLearner',
    'model_path': './model/SLearner_2pos_lr3_alpha=0.1',
    'last_model_path': '',
    'train_data': '../data/criteo_train.csv', 
    'val_data': '../data/criteo_val.csv',
    'batch_size': 256,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'summary_steps': 1000,
    'clipnorm': 5e3,
    'first_decay_steps': 1000,
}

parser = argparse.ArgumentParser(description='Train a model for Criteo dataset.')
parser.add_argument('--model_class_name', type=str, default=config['model_class_name'],
                    help='The name of the model class to train.')
parser.add_argument('--model_path', type=str, default=config['model_path'],
                    help='The path to save the model and logs.')
parser.add_argument('--clipnorm', type=float, default=5e3, help='Gradient clipnorm')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')


args = parser.parse_args()

# 使用命令行参数更新 config 字典
config['model_class_name'] = args.model_class_name
config['model_path'] = args.model_path
config['clipnorm'] = args.clipnorm
config['learning_rate'] = args.lr



print("--- 运行配置 ---")
print(f"Model Class: {config['model_class_name']}")
print(f"Model Path: {config['model_path']}")
print(f"clipnorm: {config['clipnorm']}")
print(f"learning rate: {config['learning_rate']}")
print("--------------------")

# In[9]:

# --- Main execution block for demonstration ---
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print(f'策略中的副本数: {strategy.num_replicas_in_sync}')
global_batch_size = config['batch_size'] * strategy.num_replicas_in_sync

# import tensorflow_addons as tfa

# 在策略范围内创建模型和优化器
with strategy.scope():
    # 从配置中动态获取并实例化模型类
    model_class = globals()[config['model_class_name']]
    model = model_class()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipnorm=config['clipnorm'])
    #optimizer = tfa.optimizers.AdamW(learning_rate=config['learning_rate'], weight_decay=1e-4, clipnorm=5e3)    
    # 学习率调度器：带 Warmup 的余弦退火
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        config['learning_rate'],
        config['first_decay_steps'],
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.01  # 最小学习率是初始值的 1%
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config['clipnorm'])
    model.compile(
        optimizer=optimizer,loss=None
    )


# In[10]:


# 加载检查点
checkpoint_dir = os.path.join(config['model_path'], 'checkpoints')
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
        # if name == '_treatment_index':
        #     # 反转：0 -> 1, 1 -> 0
        #     reversed_value = tf.cast(1 - value, tf.int32)
        #     labels[name] = reversed_value
        # else:
        labels[name] = value

    return features, labels  # 返回 (features, labels) 其中 labels 是 dict
# --- 应用 map 转换 ---
train_samples = train_samples.map(
    _to_features_labels,
    num_parallel_calls=4  # ✅ 安全值，不要用 AUTOTUNE
)  # ✅ 避免缓冲区过大


val_samples = dataset.prepare_dataset(config['val_data'], phase='test', batch_size=global_batch_size, shuffle=False)
# --- 应用 map 转换 ---
val_samples = val_samples.map(
    _to_features_labels,
    num_parallel_calls=4  # ✅ 安全值，不要用 AUTOTUNE
).prefetch(1)  # ✅ 避免缓冲区过大


# In[ ]:


model.fit(train_samples, validation_data=val_samples,steps_per_epoch=500, epochs=config['num_epochs'], callbacks=callbacks)

# 保存最终模型
print(f"训练完成，正在将模型保存到: {config['model_path']}")
model.save(config['model_path'])