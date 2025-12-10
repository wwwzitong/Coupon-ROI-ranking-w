from __future__ import print_function, absolute_import, division
import os
import tensorflow as tf
import numpy as np
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from data_utils import *
from ecom_dfcl import EcomDFCL_v3
from ecom_drm import EcomDRM19
from ecom_dfl import EcomDFL
from ecom_slearner import SLearner
from base_NNmodel import basemodel
from base_DFCL import base_DFCL
from dfcl_regretNet_v1_rplusc import EcomDFCL_regretNet_rplusc
from dfcl_regretNet_v1_rc import EcomDFCL_regretNet_rc
from dfcl_regretNet_v2_tau import EcomDFCL_regretNet_tau
import argparse # 导入 argparse 模块
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有 GPU，自然不会加载 CUDA。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息（隐藏 INFO 和 WARNING）

# In[7]:


# ==================== 建议添加的回调 ====================
class AUCCCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir, targets, treatment_order):
        super().__init__()
        self.validation_data = validation_data # 传入一个 tf.data.Dataset
        self.file_writer = tf.summary.create_file_writer(log_dir + '/metrics')
        self.targets = targets
        self.treatment_order = treatment_order

    def on_epoch_end(self, epoch, logs=None):
        # 从Dataset中获取一批或全部验证数据
        # 注意：如果验证集很大，应只取一个或几个batch进行估算，或对数据进行采样
        features, labels = self.validation_data
        
        predictions = self.model(features, training=False)
        
        treatment_true = labels['_treatment_index'].numpy()

        with self.file_writer.as_default():
            for target in self.targets:
                uplift_pred = (predictions[f'{target}_treatment_{self.treatment_order[0]}'] - 
                               predictions[f'{target}_treatment_{self.treatment_order[1]}']).numpy()
                
                # uplift_auc_score 需要二分类的 outcome
                # 这里以 outcome > 0 为正例，你可以根据业务逻辑调整
                outcome_true = (labels[target].numpy() > 0).astype(int)

                # # 使用评估库计算AUCC
                # aucc = uplift_auc_score(
                #     y_true=outcome_true,
                #     uplift=uplift_pred,
                #     treatment=treatment_true
                # )
                
                # 使用简化的Qini AUC计算作为替代
                aucc = self.calculate_qini_auc(uplift_pred, treatment_true, outcome_true)

                print(f'\nEpoch {epoch+1}: validation AUCC ({target}) = {aucc:.4f}')
                tf.summary.scalar(f'epoch_aucc_{target}', data=aucc, step=epoch)

    def calculate_qini_auc(self, uplift, treatment, outcome):
        """一个简化的Qini AUC计算实现"""
        order = np.argsort(uplift)[::-1]
        uplift, treatment, outcome = uplift[order], treatment[order], outcome[order]

        n = len(uplift)
        n_t = np.sum(treatment)
        n_c = n - n_t

        if n_t == 0 or n_c == 0:
            return 0.0

        y_t_cum = np.cumsum(outcome * treatment)
        y_c_cum = np.cumsum(outcome * (1 - treatment))
        
        qini_curve = (y_t_cum / n_t) - (y_c_cum / n_c)
        
        x_axis = np.arange(1, n + 1) / n
        return np.trapz(qini_curve, x_axis)

# In[8]:


# --- 1. 配置字典（替代命令行参数） ---
config = {
    'model_class_name': 'EcomDFCL_v3',
    'model_path': './model/ecom_DFCL_3entropy_2pos_lr4',
    'last_model_path': '',
    'train_data': 'data/criteo_train.csv', 
    'val_data': 'data/criteo_val.csv',
    'batch_size': 256,
    'num_epochs': 50,
    'learning_rate': 0.0001,
    'summary_steps': 1000,
}

# --- 1b. 使用 argparse 解析命令行参数 ---
parser = argparse.ArgumentParser(description='Train a model for Criteo dataset.')
parser.add_argument('--model_class_name', type=str, default=config['model_class_name'],
                    help='The name of the model class to train.')
parser.add_argument('--model_path', type=str, default=config['model_path'],
                    help='The path to save the model and logs.')
parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for the loss function.')
args = parser.parse_args()

# 使用命令行参数更新 config 字典
config['model_class_name'] = args.model_class_name
config['model_path'] = args.model_path
config['alpha'] = args.alpha

print("--- 运行配置 ---")
print(f"Model Class: {config['model_class_name']}")
print(f"Model Path: {config['model_path']}")
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
    # 将 alpha 传入模型构造函数
    # model = model_class(alpha=config['alpha'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipnorm=5e3)
    # optimizer = tfa.optimizers.AdamW(learning_rate=config['learning_rate'], weight_decay=1e-4, clipnorm=5e3)    
    model.compile(
        optimizer=optimizer,loss=None
    )




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
# aucc_callback = AUCCCallback(
#     validation_data=val_samples.batch(8192), # 比如用一个大的batch
#     log_dir=os.path.join(config['model_path'], 'logs'),
#     targets=model.targets,
#     treatment_order=model.treatment_order
# )
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
    # aucc_callback,
]

# 去掉了steps_per_epoch = 300, 
model.fit(train_samples, validation_data=val_samples, epochs=config['num_epochs'], steps_per_epoch = 500, callbacks=callbacks) # ,verbose=2) # 只在每个 epoch 结束后打印一行日志

# 保存最终模型
print(f"训练完成，正在将模型保存到: {config['model_path']}")
model.save(config['model_path'])