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

class EcomDFL(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = 99.71/(100-99.71)

        self.sparse_feature_names = SPARSE_FEATURE_NAME
        self.dense_feature_names = DENSE_FEATURE_NAME
        self.sparse_feature_slot_map = SPARSE_FEATURE_NAME_SLOT_ID
        self.num_estimated_vec_features = 10000
        self.sparse_feature_dim = 8
        self.dense_feature_dim = 1
        self.treatment_order = [1, 0]  # 处理组为15off，另一组是空白组
        self.targets = ['conversion', 'visit']


        # ✅ 新增：单一 ROI 预测头
        self.roi_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, name='roi_output')  # 输出 logit
        ], name='roi_head')

        # 特征处理层
        self._build_feature_layers()

        # 用户塔（共享底层）
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal')
        ], name='user_tower')
        
        # =======9-5Additional，为了能够保存中间activation=========
         # 显式构建 user_tower 以提前创建权重
        user_tower_input_dim = (
            len(self.sparse_feature_names) * self.sparse_feature_dim +
            len(self.dense_feature_names) * self.dense_feature_dim
        )
        self.user_tower.build(input_shape=(None, user_tower_input_dim))
        
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
        # for feature_name in self.sparse_feature_names:
        #     slot_id = self.sparse_feature_slot_map.get(feature_name)
        #     if slot_id:
        #         feature_input = inputs[feature_name]
        #         # sparse_vectors.append(self.embedding_layers[slot_id](feature_input))
        #         # 步骤 1: 将字符串输入通过 Hashing 层转换为整数 ID
        #         hashed_ids = self.hashing_layers[feature_name](feature_input)
                
        #         # 步骤 2: 将整数 ID 送入标准的 Embedding 层
        #         embedding_vec = self.embedding_layers[str(slot_id)](hashed_ids)
                
        #         sparse_vectors.append(embedding_vec)

        dense_vectors = []
        for feature_name in self.dense_feature_names:
            fcd = inputs[feature_name]
            fcd = tf.math.log1p(tf.maximum(fcd, 0.0))
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

        # 只输出一个 ROI 预测
        roi_logits = self.roi_head(shared_output, training=training)# eq3.1
        roi_score = tf.nn.sigmoid(roi_logits)  # eq3.2 q_i = σ(s_i)

        return {'score': roi_score, 'roi_logits': roi_logits}, shared_output, user_tower_activations
    
    def _add_summaries(self, name, tensor, step):
        """辅助函数，用于在TensorBoard中记录张量的统计信息"""
        tf.summary.scalar(f"{name}/mean", tf.reduce_mean(tensor), step=step)
        tf.summary.scalar(f"{name}/max", tf.reduce_max(tensor), step=step)
        tf.summary.scalar(f"{name}/min", tf.reduce_min(tensor), step=step)
        tf.summary.histogram(f"{name}/histogram", tensor, step=step)
    
    def train_step(self, data): #全量样本 或者换成batch足够大
        features, labels = data

        with tf.GradientTape() as tape:
            # 前向传播
            predictions, shared_output, user_tower_activations = self(features, training=True)
            roi_score = predictions['score']  # shape: [batch_size]
            roi_logits = predictions['roi_logits']
            # --- 监控与调试: 主动错误检测 ---
            roi_score = tf.debugging.check_numerics(roi_score, "NaN/Inf in roi_score prediction")
            roi_logits = tf.debugging.check_numerics(roi_logits, "NaN/Inf in roi_logits prediction")

            # 获取真实值
            conversion = labels['conversion']  # shape: [batch_size]
            visit = labels['visit']  # shape: [batch_size]
            treatment = labels['treatment']  # shape: [batch_size], 1 或 0

             # ✅ 1. 定义正样本并创建样本权重以应对数据稀疏问题
            # 正样本是那些产生了购买行为的样本 (gmv > 0)
            is_positive = conversion > 0
            sample_weights = tf.where(is_positive, self.pos_weight, 1.0)

            # 分组：treatment == 15 vs treatment == 0
            treatment_mask_15 = tf.equal(treatment, 1)
            treatment_mask_0 = tf.equal(treatment, 0)

            # 获取N1, N0
            n1 = statistical_config["N1"]
            n0 = statistical_config["N0"]

            # 提取对应样本的 roi_logits, conversion, visit
            roi_15 = tf.boolean_mask(roi_score, treatment_mask_15)
            roi_logits_15 = tf.boolean_mask(roi_logits, treatment_mask_15)
            conversion_15 = tf.boolean_mask(conversion, treatment_mask_15)
            visit_15 = tf.boolean_mask(visit, treatment_mask_15)
            weights_15 = tf.boolean_mask(sample_weights, treatment_mask_15) # ✅ 提取对应样本的权重

            roi_0 = tf.boolean_mask(roi_score, treatment_mask_0)
            roi_logits_0 = tf.boolean_mask(roi_logits, treatment_mask_0)
            conversion_0 = tf.boolean_mask(conversion, treatment_mask_0)
            visit_0 = tf.boolean_mask(visit, treatment_mask_0)
            weights_0 = tf.boolean_mask(sample_weights, treatment_mask_0) # ✅ 提取对应样本的权重

            # ✅ 根据你提供的公式更新损失计算
            # y_r * ln(q/(1-q)) + y_c * ln(1-q)
            # 其中 ln(q/(1-q)) 等于 logit，ln(1-q) 等于 log_sigmoid(-logit)
            # loss_15 = tf.reduce_sum(
            #     conversion_15 * roi_logits_15 +
            #     visit_15 * (-tf.nn.softplus(roi_logits_15))
            # )
            loss_per_sample_15 = conversion_15 * roi_logits_15 + visit_15 * (-tf.nn.softplus(roi_logits_15))
            loss_15 = tf.reduce_sum(loss_per_sample_15 * weights_15)

            # loss_0 = tf.reduce_sum(
            #     conversion_0 * roi_logits_0 +
            #     visit_0 * -tf.nn.softplus(roi_logits_0)
            # )
            loss_per_sample_0 = conversion_0 * roi_logits_0 + visit_0 * (-tf.nn.softplus(roi_logits_0))
            loss_0 = tf.reduce_sum(loss_per_sample_0 * weights_0)

            # 公式 (3): -[ (1/N1)*loss_15 - (1/N0)*loss_0 ] 加个常数防止loss太小了
            total_loss = -10*(loss_15 / n1 - loss_0 / n0)
            
            # --- 监控与调试: 检查最终loss ---
            total_loss = tf.debugging.check_numerics(total_loss, "NaN/Inf in total_loss")

            # ✅ 注意：这里不是加权 task loss，而是单一 ROI loss
            # 如果你想加正则项，可额外加入：
            # total_loss += 1e-4 * tf.reduce_mean(tf.square(self.trainable_variables))

        # 反向传播
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # --- 全面监控记录到 TensorBoard ---
        step = self.optimizer.iterations

        # 思路1: 监控前向传播
        self._add_summaries("labels/conversion", conversion, step=step)
        self._add_summaries("labels/visit", visit, step=step)
        self._add_summaries("predictions/roi_score", roi_score, step=step)
        self._add_summaries("activations/shared_output", shared_output, step=step)

        # 监控 User Tower 的每一层激活
        for name, activation in user_tower_activations.items():
            self._add_summaries(f"activations/user_tower/{name}", activation, step=step)

        # 监控损失分量
        tf.summary.scalar("losses/total_loss", total_loss, step=step)
        tf.summary.scalar("losses/loss_15_avg", loss_15 / n1, step=step)
        tf.summary.scalar("losses/loss_0_avg", loss_0 / n0, step=step)

        # 监控中间计算值
        tf.summary.scalar("calculation/n1", n1, step=step)
        tf.summary.scalar("calculation/n0", n0, step=step)
        self._add_summaries("calculation/roi_logits_15", roi_logits_15, step=step)
        self._add_summaries("calculation/roi_logits_0", roi_logits_0, step=step)

        # 监控 Embedding 范数 (假设模型有 embedding_layers 属性)
        if hasattr(self, 'embedding_layers'):
            for slot_id, emb_layer in self.embedding_layers.items():
                embedding_norm = tf.norm(emb_layer.weights[0], axis=1)
                self._add_summaries(f"embeddings/slot_{slot_id}_norm", embedding_norm, step=step)

        # 监控坏样本 (可选，会影响性能)
        if 'user_id' in features:
            tf.cond(
                total_loss > 1000.0,  # 可根据实际情况调整阈值
                lambda: tf.print("High loss detected! Loss:", total_loss, "UserIDs:", features['user_id'][:5], summarize=-1),
                lambda: tf.constant(0) # 空操作
            )

        # 思路2: 监控梯度
        valid_gradients = [g for g in gradients if g is not None]
        if valid_gradients:
            global_norm = tf.linalg.global_norm(valid_gradients)
            tf.summary.scalar("gradients/global_norm", global_norm, step=step)
            # 抽样监控几个关键层的梯度
            for i, grad in enumerate(valid_gradients):
                if i % 10 == 0: # 每隔10个变量记录一次，避免日志过大
                    self._add_summaries(f"gradients/var_{i}", grad, step=step)
        # --- 调试结束 ---

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "total_loss": total_loss,
            "loss_15": loss_15 / n1,
            "loss_0": loss_0 / n0,
            "roi_mean_15": tf.reduce_mean(roi_15),
            "roi_mean_0": tf.reduce_mean(roi_0),
        }

    def test_step(self, data): #
        features, labels = data

        # 前向传播
        predictions = self(features, training=False)
        roi_score = predictions['score']
        roi_logits = predictions['roi_logits']

        # 获取真实值
        conversion = labels['conversion']
        visit = labels['visit']
        treatment = labels['_treatment_index']

        # ✅ 1. 定义正样本并创建样本权重
        is_positive = conversion > 0
        sample_weights = tf.where(is_positive, self.pos_weight, 1.0)
        
        # 分组
        treatment_mask_15 = tf.equal(treatment, 1)
        treatment_mask_0 = tf.equal(treatment, 0)

        # 获取N1, N0
        n1 = tf.cast(statistical_config["N1"], tf.float32)
        n0 = tf.cast(statistical_config["N0"], tf.float32)

        # 提取对应样本
        roi_15 = tf.boolean_mask(roi_score, treatment_mask_15)
        roi_logits_15 = tf.boolean_mask(roi_logits, treatment_mask_15)
        conversion_15 = tf.boolean_mask(conversion, treatment_mask_15)
        visit_15 = tf.boolean_mask(visit, treatment_mask_15)
        weights_15 = tf.boolean_mask(sample_weights, treatment_mask_15)

        roi_0 = tf.boolean_mask(roi_score, treatment_mask_0)
        roi_logits_0 = tf.boolean_mask(roi_logits, treatment_mask_0)
        conversion_0 = tf.boolean_mask(conversion, treatment_mask_0)
        visit_0 = tf.boolean_mask(visit, treatment_mask_0)
        weights_0 = tf.boolean_mask(sample_weights, treatment_mask_0)

        # 计算损失
        loss_per_sample_15 = conversion_15 * roi_logits_15 + visit_15 * (-tf.nn.softplus(roi_logits_15))
        loss_15 = tf.reduce_sum(loss_per_sample_15 * weights_15)
    
        loss_per_sample_0 = conversion_0 * roi_logits_0 + visit_0 * (-tf.nn.softplus(roi_logits_0))
        loss_0 = tf.reduce_sum(loss_per_sample_0 * weights_0)

        total_loss = -10 * (loss_15 / n1 - loss_0 / n0)

        # 返回 Keras 用于日志记录的指标
        return {
            "total_loss": total_loss,
            "loss_15": loss_15 / n1,
            "loss_0": loss_0 / n0,
            "roi_mean_15": tf.reduce_mean(roi_15),
            "roi_mean_0": tf.reduce_mean(roi_0),
        }