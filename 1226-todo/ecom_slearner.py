import tensorflow as tf
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
import os
import sys
import io
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_utils import *
SPARSE_FEATURE_NAME = []
DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
SPARSE_FEATURE_NAME_SLOT_ID = {}
statistical_config={
    'N':11183673,
    'N1':9506123,
    'N0':1677550
}

# # 强制UTF-8编码
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# os.environ['PYTHONIOENCODING'] = 'utf-8'

class SLearner(tf.keras.Model): # std+2pos
    """
    使用 TensorFlow 2.x Keras API 实现的电商模型。
    该模型集成了 GradNorm 和自定义决策损失。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paid_pos_weight = 99.71/(100-99.71)
        self.cost_pos_weight=95.30/(100-95.30)
        
        # 从 fsfc.py 导入配置
        self.sparse_feature_names = SPARSE_FEATURE_NAME
        self.dense_feature_names = DENSE_FEATURE_NAME
        self.sparse_feature_slot_map = SPARSE_FEATURE_NAME_SLOT_ID
        self.num_estimated_vec_features = 10000# 为什么要这么大数量 12000000

        # 模型超参数 TODO：需要依据实际数据集进行修改
        self.sparse_feature_dim = 8 # TODO：简化为4
        self.dense_feature_dim = 1
        self.treatment_order = [1, 0] #处理组为15off，另一组是空白组
#         self.ratios = [0.1, 0.5, 1.0]  #先用少量测试
        self.ratios = [i / 100.0 for i in range(5, 105, 5)] #ratio也就是lambda，这里应该换成更为密集的，真正模拟积分。
        self.targets = ['paid', 'cost']
        
        self.total_samples = statistical_config['N']
        self.treatment_sample_counts = {
            '1': statistical_config['N1'],
            '0': statistical_config['N0']
        }
        
        # 特征处理层
        self._build_feature_layers()
        
        # 网络结构，user_tower为共享底层
        # self.user_tower = tf.keras.Sequential([
        #     tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'), # 512->128
        #     tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),# 256->64
        #     tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal')# 128->32
        # ], name='user_tower')
        # TOWER
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'), # 512->128
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),# 256->64
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal')# 128->32
        ], name='user_tower')
        
        # =======9-5Additional，为了能够保存中间activation=========
         # 显式构建 user_tower 以提前创建权重
        user_tower_input_dim = (
            len(self.sparse_feature_names) * self.sparse_feature_dim +
            len(self.dense_feature_names) * self.dense_feature_dim
        )
        self.user_tower.build(input_shape=(None, user_tower_input_dim))
        
        self.task_towers = {} # 2x2，为prediction loss服务
        tower_dims = [64, 32, 1]
        for target in self.targets:
            for treatment in self.treatment_order:
                name = "{}_treatment_{}_tower".format(target, treatment)
                self.task_towers[name] = tf.keras.Sequential([
                    tf.keras.layers.Dense(dims, activation='relu', kernel_initializer='glorot_normal') for dims in tower_dims[:-1]
                ] + [
                    tf.keras.layers.Dense(tower_dims[-1], kernel_initializer='glorot_normal')
                ], name=name)

        # GradNorm 的可训练损失权重
#         self.loss_weights = tf.Variable([1.0, 1.0], trainable=True, name='grad_norm_loss_weights')

    def _build_feature_layers(self):# 特征embedding层，为call部分的sparse features服务
        # 3. 修改此方法，同时创建 Hashing 层和 Embedding 层
        # Hashing 层: 每个稀疏特征对应一个，负责 string -> int
        self.hashing_layers = {}
        for feature_name in self.sparse_feature_names:
            self.hashing_layers[feature_name] = tf.keras.layers.Hashing(
                num_bins=self.num_estimated_vec_features, 
                name="hashing_{}".format(feature_name)
            )

        self.embedding_layers = {}
        unique_slot_ids = set(self.sparse_feature_slot_map.values())
        for slot_id in unique_slot_ids:
            self.embedding_layers[str(slot_id)] = tf.keras.layers.Embedding(
                input_dim=self.num_estimated_vec_features,
                output_dim=self.sparse_feature_dim,
                name="embedding_slot_{}".format(slot_id)
            )

    def call(self, inputs, training=True): # 定义数据从输入到输出的完整流动路径       
        # 特征处理
        sparse_vectors = []
        dense_vectors = []
        for feature_name in self.dense_feature_names:
            fcd = inputs[feature_name]
            fcd = tf.math.log1p(tf.maximum(fcd,0.0))
            fcd = (fcd - tf.reduce_mean(fcd)) / (tf.math.reduce_std(fcd) + 1e-8)
            dense_vectors.append(tf.reshape(fcd, [-1, self.dense_feature_dim]))

        concat_input = tf.concat(sparse_vectors + dense_vectors, axis=1)
        
#         shared_output = self.user_tower(concat_input, training=training)
        # 为了监控中间层激活，手动执行 user_tower 的前向传播
        x = concat_input
        user_tower_activations = {}
        for i, layer in enumerate(self.user_tower.layers):
            x = layer(x, training=training)
            # 使用一个在 tf.function 中唯一的名称
            user_tower_activations[f"layer_{i}_{layer.name}"] = x
        shared_output = x
         
        predictions = {}
        for name, tower in self.task_towers.items():
            # name is like "paid_treatment_30_tower"
            pred_name = name.replace('_tower', '')
            logit = tower(shared_output, training=training)
            predictions[pred_name] = tf.reshape(logit, [-1])
                
        self._last_shared_output = shared_output
        self._last_user_tower_activations = user_tower_activations
        return predictions
    
    def compute_local_losses(self, predictions, labels): #和论文不一致啊
        paid_loss = tf.constant(0.0, dtype=tf.float32)
        cost_loss = tf.constant(0.0, dtype=tf.float32)

        # 预先计算 treatment_mask（只一次）
        treatment_idx = tf.cast(labels['treatment'], tf.int32)

        for target_name in self.targets:
            if target_name == 'paid':
                pos_weight = self.paid_pos_weight
            else:
                pos_weight = self.cost_pos_weight
            local_loss = tf.constant(0.0, dtype=tf.float32)
            for treatment in self.treatment_order:
                pred_name = f"{target_name}_treatment_{treatment}"
                logit = predictions[pred_name]

                # ✅ 一次性处理：避免多次调用 tf.minimum、tf.abs 等
                out = tf.minimum(logit, 10.0)
                label = tf.cast(labels[target_name], tf.float32) 
                # label = labels[target_name]

                # ✅ 使用广播和向量化计算，避免循环
                term1 = -label * out
                term2 = (1 + label) * (tf.maximum(out, 0) + tf.math.log(1 + tf.exp(-tf.abs(out))))
                loss_per_sample = term1 + term2

                # ✅ mask 只构造一次
                treatment_mask = tf.cast(tf.equal(treatment_idx, treatment), tf.float32)
                # masked_loss = loss_per_sample * treatment_mask  # 直接乘，更高效
                # 【核心修改】根据 pos_weight 对正样本的损失进行加权.当 label > 0 时，权重为 pos_weight，否则为 1.0

                sample_weights = tf.where(label > 0, pos_weight, 1.0)
                weighted_loss_per_sample = loss_per_sample * sample_weights
                masked_loss = weighted_loss_per_sample * treatment_mask  # 使用加权后的损失
                
                # ✅ 累加 sum
                local_loss += tf.reduce_sum(masked_loss)

            if target_name == 'paid':
                paid_loss += local_loss
            else:
                cost_loss += local_loss

        return paid_loss, cost_loss
    
    def _add_summaries(self, name, tensor, step):
        """辅助函数，用于在TensorBoard中记录张量的统计信息"""
        tf.summary.scalar(f"{name}/mean", tf.reduce_mean(tensor), step=step)
        tf.summary.scalar(f"{name}/max", tf.reduce_max(tensor), step=step)
        tf.summary.scalar(f"{name}/min", tf.reduce_min(tensor), step=step)
        tf.summary.histogram(f"{name}/histogram", tensor, step=step)

    def train_step(self, data):
    
        features, labels = data

        with tf.GradientTape(persistent=True) as tape:
            # 1. 前向传播，只获取 predictions
            predictions = self(features, training=True)
            
            # 从模型属性中获取中间激活值
            shared_output = self._last_shared_output
            user_tower_activations = self._last_user_tower_activations
            
            # --- 进阶监控 3: 主动错误检测 ---
            for name, pred in predictions.items():
                predictions[name] = tf.debugging.check_numerics(pred, f"NaN/Inf in prediction: {name}")

            # 将模型的所有可训练变量分为两组：
            # 1. 模型参数（如 Tower、Embedding 等）
            # 2. GradNorm 的损失权重
            model_variables = [v for v in self.trainable_variables if 'loss_weights' not in v.name]

            paid_loss, cost_loss = self.compute_local_losses(predictions, labels)
            
            # --- 进阶监控 3 (续): 检查最终loss ---
            paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in paid_loss")
            cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in cost_loss")
            
            # 2. 计算用于更新【模型参数】的损失
            weighted_task_loss = 0.5 * paid_loss + 0.5 * cost_loss
            # 对应您代码中的 total_loss
            model_update_loss = weighted_task_loss * len(self.targets)
        

        # 4. 在 tape 上下文之外，分别计算两部分梯度
        # 计算模型参数的梯度
        model_gradients = tape.gradient(model_update_loss, model_variables)
                
        # Add
        self.optimizer.apply_gradients(zip(model_gradients, self.trainable_variables))
        
        # --- 全面监控记录到 TensorBoard ---
        step = self.optimizer.iterations
        # 思路1: 监控前向传播
        self._add_summaries("labels/paid", labels['paid'], step=step)
        self._add_summaries("labels/cost", labels['cost'], step=step)
        self._add_summaries("activations/shared_output",  shared_output, step=step)

        # 监控损失分量
        self._add_summaries("losses/1_paid_loss", paid_loss, step=step)
        self._add_summaries("losses/2_cost_loss", cost_loss, step=step)
        self._add_summaries("losses/3_weighted_task_loss", weighted_task_loss, step=step)
        self._add_summaries("losses/5_model_update_loss", model_update_loss, step=step)
        
        # --- 进阶监控 1: 逐层激活与预测值 ---
        # 监控 User Tower 的每一层激活
        for name, activation in user_tower_activations.items():
            self._add_summaries(f"activations/user_tower/{name}", activation, step=step)
            
        # 监控 Task Towers 的 logits
        for name, pred_logit in predictions.items():
            self._add_summaries(f"predictions/{name}_logit", pred_logit, step=step)

        # --- 进阶监控 2: Embedding范数 ---
        for slot_id, emb_layer in self.embedding_layers.items():
            # emb_layer.weights[0] 是 embedding 矩阵
            embedding_norm = tf.norm(emb_layer.weights[0], axis=1)
            self._add_summaries(f"embeddings/slot_{slot_id}_norm", embedding_norm, step=step)
                
        # --- 进阶监控 4: 定位坏样本 (可选，会影响性能) ---
        # 当 loss 超过一个阈值时，打印 user_id
        # 注意: tf.print 在 graph 模式下需要配合 tf.cond
        if 'id' in features:
            tf.cond(
                model_update_loss > 1000.0,  # 设置一个较高的阈值
                lambda: tf.print("High loss detected! Loss:", model_update_loss, "UserIDs:", features['id'][:5], summarize=-1),
                lambda: tf.constant(0) # 空操作
            )

        # 思路2: 监控梯度
        # 过滤掉None的梯度
        valid_gradients = [g for g in model_gradients if g is not None]
        if valid_gradients:
            global_norm = tf.linalg.global_norm(valid_gradients)
            tf.summary.scalar("gradients/global_norm", global_norm, step=step)
            # 抽样监控几个关键层的梯度
            for i, grad in enumerate(valid_gradients):
                if i % 10 == 0: # 每隔10个变量记录一次，避免日志过大
                    self._add_summaries(f"gradients/var_{i}", grad, step=step)
        # --- 调试结束 ---

        # 6. persistent=True 时，手动删除 tape 释放资源
        del tape

        return {
            "total_loss": model_update_loss, 
            "weighted_task_loss": weighted_task_loss, 
            "paid_loss": paid_loss, "cost_loss": cost_loss,
        }

    def test_step(self, data):
        """验证步骤，不更新权重，只计算损失"""
        features, labels = data
        
       # --- 终极调试：检查 Softmax 的行为 ---
        # tf.print("\n--- 调试 val_step ---")
        # 1. 检查 treatment 和标签
        # tf.print("样本 treatment:", labels['treatment'][:5], summarize=-1)
        # tf.print("paid 标签总和:", tf.reduce_sum(labels['paid']))

        predictions = self(features, training=False)

        # --- 为了调试，在这里重新计算 decision_loss 的关键部分并打印 ---
        pred_dict = {key: tf.exp(tf.minimum(logit, 10.0)) for key, logit in predictions.items()}
        tau = self.tau if hasattr(self, 'tau') else 1.1
        ratio = self.ratios[0] # 只看第一个 ratio 的情况来简化调试
        
        values = [pred_dict[f"paid_treatment_{t}"] - ratio * pred_dict[f"cost_treatment_{t}"] for t in self.treatment_order]
        cancat_tensor = tf.stack(values, axis=1)
        logits_before_softmax = cancat_tensor / tau
        softmax_tensor = tf.nn.softmax(logits_before_softmax, axis=1)

        treatment_idx = tf.cast(labels['treatment'], tf.int32)
        mask_list = [tf.reshape(tf.cast(tf.equal(t, treatment_idx), tf.float32), [-1, 1]) for t in self.treatment_order]
        mask_tensor = tf.concat(mask_list, axis=1)
        
        # 2. 检查 mask_tensor (已知是正确的，但保留以作对比)
        # tf.print("样本 mask_tensor:", mask_tensor[:5], summarize=-1)

        # 3. 【关键】检查 softmax 的输入和输出
        # tf.print("【关键】Softmax 输入 (Logits):", logits_before_softmax[:5], summarize=-1)
        # tf.print("【关键】Softmax 输出 (Probs):", softmax_tensor[:5], summarize=-1)

        # 4. 将它们与 mask 相乘，查看结果
        masked_softmax = softmax_tensor * mask_tensor
        # tf.print("【关键】Masked Softmax Probs:", masked_softmax[:5], summarize=-1)
        # tf.print("【关键】Masked Softmax Probs (总和):", tf.reduce_sum(masked_softmax))
        
        # tf.print("--- 调试结束 ---")
        # --- 调试代码结束 ---
        
        for name, pred in predictions.items():
            predictions[name] = tf.debugging.check_numerics(pred, f"NaN/Inf in validation prediction: {name}")

        paid_loss, cost_loss = self.compute_local_losses(predictions, labels)
        
        paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in val_paid_loss")
        cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in val_cost_loss")

        weighted_task_loss = 0.5 * paid_loss + 0.5 * cost_loss
        model_update_loss = weighted_task_loss * len(self.targets)

        # 移除返回字典键中的 "val_" 前缀，Keras 会自动添加
        return {
            "total_loss": model_update_loss,
            "weighted_task_loss": weighted_task_loss, 
            "paid_loss": paid_loss, 
            "cost_loss": cost_loss,
        }