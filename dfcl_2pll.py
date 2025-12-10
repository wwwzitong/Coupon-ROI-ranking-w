import tensorflow as tf
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from data_utils import *
SPARSE_FEATURE_NAME = []
DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
SPARSE_FEATURE_NAME_SLOT_ID = {}
statistical_config={
    'N':11183673,
    'N1':9506123,
    'N0':1677550
}

class EcomDFCL_v3(tf.keras.Model): # std+ifdl一系列clip、maximum+2pos
    """
    decision_policy_learning_loss 2pos+std+tower
    """
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.conversion_pos_weight = 99.71/(100-99.71)
        self.visit_pos_weight=95.30/(100-95.30)
        self.alpha = alpha #prediction loss前面的系数
        
        # 从 fsfc.py 导入配置
        self.sparse_feature_names = SPARSE_FEATURE_NAME
        self.dense_feature_names = DENSE_FEATURE_NAME
        self.sparse_feature_slot_map = SPARSE_FEATURE_NAME_SLOT_ID
        self.num_estimated_vec_features = 10000

        # 模型超参数 TODO：需要依据实际数据集进行修改
        self.sparse_feature_dim = 8 # TODO：简化为4
        self.dense_feature_dim = 1
        self.treatment_order = [1, 0] #处理组为15off，另一组是空白组
#         self.ratios = [0.1, 0.5, 1.0]  #先用少量测试
        self.ratios = [i / 100.0 for i in range(5, 105, 5)] #ratio也就是lambda，这里应该换成更为密集的，真正模拟积分。
        self.targets = ['conversion', 'visit']
        
        self.total_samples = statistical_config['N']
        self.treatment_sample_counts = {
            1: statistical_config['N1'],
            0: statistical_config['N0']
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
            # name is like "conversion_treatment_30_tower"
            pred_name = name.replace('_tower', '')
            logit = tower(shared_output, training=training)
            predictions[pred_name] = tf.reshape(logit, [-1])
                
        self._last_shared_output = shared_output
        self._last_user_tower_activations = user_tower_activations
        return predictions
    
    def compute_local_losses(self, predictions, labels): #和论文不一致啊
        conversion_loss = tf.constant(0.0, dtype=tf.float32)
        visit_loss = tf.constant(0.0, dtype=tf.float32)

        # 预先计算 treatment_mask（只一次）
        treatment_idx = tf.cast(labels['treatment'], tf.int32)

        for target_name in self.targets:
            if target_name == 'conversion':
                pos_weight = self.conversion_pos_weight
            else:
                pos_weight = self.visit_pos_weight
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

            if target_name == 'conversion':
                conversion_loss += local_loss
            else:
                visit_loss += local_loss

        return conversion_loss, visit_loss

    def decision_policy_learning_loss(self, predictions, labels): #decision loss也和论文有出入
        '''
        预测层只有prediction loss相关的，没有和决策直接相关的.
        很奇怪。这里相当于将prediction loss相关变量组合为决策预测层
        '''
        pred_dict = {key: tf.exp(tf.minimum(logit, 10.0)) for key, logit in predictions.items()}

        decision_loss_sum = tf.constant(0.0, dtype=tf.float32)

        # 预先计算 treatment_mask（仅一次）
        treatment_idx = tf.cast(labels['treatment'], tf.int32)
        treatment_masks = {
            t: tf.cast(tf.equal(treatment_idx, t), tf.float32) for t in self.treatment_order
        }
        
        # --- 修改：使用预设的 N 和 N_ti 计算权重 ---
        N = self.total_samples
        counts_per_treatment = {
            t: float(self.treatment_sample_counts[t]) + 1e-8 for t in self.treatment_order
        }
        
        # 构建权重张量, weight_tensor[i] = N / N_ti for sample i in treatment ti
        weight_tensor = tf.zeros_like(treatment_idx, dtype=tf.float32)
        for t in self.treatment_order:
            weight_for_t = N / counts_per_treatment[t]
            weight_tensor += treatment_masks[t] * weight_for_t
        
        # Reshape for broadcasting
        weight_tensor = tf.reshape(weight_tensor, [-1, 1])
        for ratio in self.ratios:
            # 使用列表推导，避免显式 append
            values = [
                pred_dict[f"conversion_treatment_{t}"] - ratio * pred_dict[f"visit_treatment_{t}"]
                for t in self.treatment_order
            ]
            cancat_tensor = tf.nn.softmax(tf.stack(values, axis=1), axis=1)

            # mask_tensor 是 one-hot 编码
            mask_tensor = tf.stack([treatment_masks[t] for t in self.treatment_order], axis=1)

            ratio_target = tf.reshape(labels['conversion'] - ratio * labels['visit'], [-1, 1])#样本的真实收益
            # cancat_tensor * mask_tensor 的结果是，只保留模型对真实 treatment 的“最优概率”预测，其他位置为0
            decision_loss = tf.reduce_sum(cancat_tensor * mask_tensor * ratio_target * weight_tensor)

            decision_loss_sum += decision_loss

        return decision_loss_sum / len(self.ratios)
    
    def _add_summaries(self, name, tensor, step):
        """辅助函数，用于在TensorBoard中记录张量的统计信息"""
        tf.summary.scalar(f"{name}/mean", tf.reduce_mean(tensor), step=step)
        tf.summary.scalar(f"{name}/max", tf.reduce_max(tensor), step=step)
        tf.summary.scalar(f"{name}/min", tf.reduce_min(tensor), step=step)
        tf.summary.histogram(f"{name}/histogram", tensor, step=step)

    def _compute_cosine_similarity(self, grads1, grads2):
        """计算两组梯度之间的余弦相似度"""
        dot_product = tf.constant(0.0, dtype=tf.float32)
        norm1_sq = tf.constant(0.0, dtype=tf.float32)
        norm2_sq = tf.constant(0.0, dtype=tf.float32)

        # 遍历梯度对，只处理两者都非None的情况
        for g1, g2 in zip(grads1, grads2):
            if g1 is not None and g2 is not None:
                dot_product += tf.reduce_sum(g1 * g2)
                norm1_sq += tf.reduce_sum(g1 * g1)
                norm2_sq += tf.reduce_sum(g2 * g2)
        
        # 避免除以零
        denominator = tf.sqrt(norm1_sq * norm2_sq)
        return tf.math.divide_no_nan(dot_product, denominator)

    def train_step(self, data):
    
        features, labels = data

        with tf.GradientTape(persistent=True) as tape:
            # 1. 前向传播，计算各个原始损失
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

            conversion_loss, visit_loss = self.compute_local_losses(predictions, labels)
            decision_loss = self.decision_policy_learning_loss(predictions, labels)
            
            # --- 进阶监控 3 (续): 检查最终loss ---
            conversion_loss = tf.debugging.check_numerics(conversion_loss, "NaN/Inf in conversion_loss")
            visit_loss = tf.debugging.check_numerics(visit_loss, "NaN/Inf in visit_loss")
            decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in decision_loss")
            
            # 2. 计算用于更新【模型参数】的损失
            weighted_task_loss = 0.5 * conversion_loss + 0.5 * visit_loss
            # 对应您代码中的 total_loss
            model_update_loss = self.alpha * weighted_task_loss * len(self.targets) - decision_loss
        

        # 4. 在 tape 上下文之外，分别计算多组梯度
        # 4.1. 计算用于更新模型的总梯度
        model_gradients = tape.gradient(model_update_loss, model_variables)
        
        # 4.2. 单独计算 task_loss 的梯度，用于分析
        task_loss_gradients = tape.gradient(weighted_task_loss, model_variables)

        # 4.3. 单独计算 decision_loss 的梯度，用于分析
        decision_loss_gradients = tape.gradient(decision_loss, model_variables)
                
        # 5. 应用总梯度更新模型参数
        # 注意：这里使用 model_variables 而不是 self.trainable_variables，以确保梯度和变量列表匹配
        self.optimizer.apply_gradients(zip(model_gradients, model_variables))
        
        # --- 梯度分析与监控 ---
        step = self.optimizer.iterations
        
        # 5.1. 计算并记录各部分梯度的范数（大小）
        # 这可以告诉我们哪个loss产生的梯度“力量”更大
        valid_task_grads = [g for g in task_loss_gradients if g is not None]
        if valid_task_grads:
            task_grad_norm = tf.linalg.global_norm(valid_task_grads)
            tf.summary.scalar("gradients_analysis/norm_task_loss", task_grad_norm, step=step)

        valid_decision_grads = [g for g in decision_loss_gradients if g is not None]
        if valid_decision_grads:
            decision_grad_norm = tf.linalg.global_norm(valid_decision_grads)
            tf.summary.scalar("gradients_analysis/norm_decision_loss", decision_grad_norm, step=step)

        # 5.2. 计算并记录 task_loss 和 decision_loss 梯度之间的余弦相似度（方向）
        # 这个指标衡量两个loss的梯度方向的对齐程度
        cosine_similarity = self._compute_cosine_similarity(task_loss_gradients, decision_loss_gradients)
        tf.summary.scalar("gradients_analysis/cosine_similarity_task_vs_decision", cosine_similarity, step=step)
        # --- 梯度分析结束 ---
        
        # 思路1: 监控前向传播
        self._add_summaries("labels/conversion", labels['conversion'], step=step)
        self._add_summaries("labels/visit", labels['visit'], step=step)
        self._add_summaries("activations/shared_output",  shared_output, step=step)

        # 监控损失分量
        self._add_summaries("losses/1_conversion_loss", conversion_loss, step=step)
        self._add_summaries("losses/2_visit_loss", visit_loss, step=step)
        self._add_summaries("losses/3_weighted_task_loss", weighted_task_loss, step=step)
        self._add_summaries("losses/4_decision_loss", decision_loss, step=step)
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
            "weighted_task_loss": weighted_task_loss, "decision_loss": decision_loss,
            "conversion_loss": conversion_loss, "visit_loss": visit_loss,
        }

    def test_step(self, data):
        """验证步骤，不更新权重，只计算损失"""
        features, labels = data
        
       # --- 终极调试：检查 Softmax 的行为 ---
        tf.print("\n--- 调试 val_step ---")
        # 1. 检查 treatment 和标签
        tf.print("样本 treatment:", labels['treatment'][:5], summarize=-1)
        tf.print("conversion 标签总和:", tf.reduce_sum(labels['conversion']))

        predictions = self(features, training=False)

        # --- 为了调试，在这里重新计算 decision_loss 的关键部分并打印 ---
        pred_dict = {key: tf.exp(tf.minimum(logit, 10.0)) for key, logit in predictions.items()}
        tau = self.tau if hasattr(self, 'tau') else 1.1
        ratio = self.ratios[0] # 只看第一个 ratio 的情况来简化调试
        
        values = [pred_dict[f"conversion_treatment_{t}"] - ratio * pred_dict[f"visit_treatment_{t}"] for t in self.treatment_order]
        cancat_tensor = tf.stack(values, axis=1)
        logits_before_softmax = cancat_tensor / tau
        softmax_tensor = tf.nn.softmax(logits_before_softmax, axis=1)

        treatment_idx = tf.cast(labels['treatment'], tf.int32)
        mask_list = [tf.reshape(tf.cast(tf.equal(t, treatment_idx), tf.float32), [-1, 1]) for t in self.treatment_order]
        mask_tensor = tf.concat(mask_list, axis=1)
        
        # 2. 检查 mask_tensor (已知是正确的，但保留以作对比)
        tf.print("样本 mask_tensor:", mask_tensor[:5], summarize=-1)

        # 3. 【关键】检查 softmax 的输入和输出
        tf.print("【关键】Softmax 输入 (Logits):", logits_before_softmax[:5], summarize=-1)
        tf.print("【关键】Softmax 输出 (Probs):", softmax_tensor[:5], summarize=-1)

        # 4. 将它们与 mask 相乘，查看结果
        masked_softmax = softmax_tensor * mask_tensor
        tf.print("【关键】Masked Softmax Probs:", masked_softmax[:5], summarize=-1)
        tf.print("【关键】Masked Softmax Probs (总和):", tf.reduce_sum(masked_softmax))
        
        tf.print("--- 调试结束 ---")
        # --- 调试代码结束 ---
        
        for name, pred in predictions.items():
            predictions[name] = tf.debugging.check_numerics(pred, f"NaN/Inf in validation prediction: {name}")

        conversion_loss, visit_loss = self.compute_local_losses(predictions, labels)
        decision_loss = self.decision_policy_learning_loss(predictions, labels)
        
        conversion_loss = tf.debugging.check_numerics(conversion_loss, "NaN/Inf in val_conversion_loss")
        visit_loss = tf.debugging.check_numerics(visit_loss, "NaN/Inf in val_visit_loss")
        decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in val_decision_loss")

        weighted_task_loss = 0.5 * conversion_loss + 0.5 * visit_loss
        model_update_loss = weighted_task_loss * len(self.targets) - decision_loss

        # 移除返回字典键中的 "val_" 前缀，Keras 会自动添加
        return {
            "total_loss": model_update_loss,
            "weighted_task_loss": weighted_task_loss, 
            "decision_loss": decision_loss,
            "conversion_loss": conversion_loss, 
            "visit_loss": visit_loss,
        }