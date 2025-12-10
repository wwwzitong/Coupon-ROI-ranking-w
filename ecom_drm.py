import tensorflow as tf
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from data_utils import *
SPARSE_FEATURE_NAME = []
DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
SPARSE_FEATURE_NAME_SLOT_ID = {}
class EcomDRM19(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = 99.71/(100-99.71)

        self.sparse_feature_names = SPARSE_FEATURE_NAME
        self.dense_feature_names = DENSE_FEATURE_NAME
        self.sparse_feature_slot_map = SPARSE_FEATURE_NAME_SLOT_ID
        self.num_estimated_vec_features = 10000
        self.sparse_feature_dim = 8
        self.dense_feature_dim = 1
        self.treatment_order = [15, 0]  # 处理组为15off，另一组是空白组
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
            
    def call(self, inputs, training=False):
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

        # ✅eq.(16) 输出原始得分 si 
        raw_scores = self.roi_head(shared_output, training=training)  # shape: [batch_size, 1]

        # eq.(17) tanh 归一化
        tanh_scores = tf.nn.tanh(raw_scores)  # 归一化到 (-1, 1)

        return {'score': tanh_scores}, shared_output, user_tower_activations

    def _add_summaries(self, name, tensor, step):
        """辅助函数，用于在TensorBoard中记录张量的统计信息"""
        tf.summary.scalar(f"{name}/mean", tf.reduce_mean(tensor), step=step)
        tf.summary.scalar(f"{name}/max", tf.reduce_max(tensor), step=step)
        tf.summary.scalar(f"{name}/min", tf.reduce_min(tensor), step=step)
        tf.summary.histogram(f"{name}/histogram", tensor, step=step)

    def train_step(self, data):
        features, labels = data

        with tf.GradientTape() as tape:
            # 前向传播
            predictions, shared_output, user_tower_activations = self(features, training=True)
            scores = predictions['score']  # shape: [batch_size]
            # --- 监控与调试: 主动错误检测 ---
            scores = tf.debugging.check_numerics(scores, "NaN/Inf in pi prediction")

            # 计算loss需要获取真实标签
            conversion = labels['conversion']
            visit = labels['visit']
            treatment = labels['treatment']  # shape: [batch_size], 1 或 0
            
            # --- 在 train_step 中进行分组 Softmax 来计算 pi ---
            treatment_mask = tf.equal(treatment, 1)
            control_mask = tf.equal(treatment, 0)
            
            # 为 tf.where 扩展 mask 维度以匹配 scores 的维度
            treatment_mask_expanded = tf.expand_dims(treatment_mask, axis=1)
            control_mask_expanded = tf.expand_dims(control_mask, axis=1)

            # 对 treatment 组计算 softmax
            # 将非本组的分数设为极小值，使其在 softmax 后概率接近0
            treatment_scores_masked = tf.where(treatment_mask_expanded, scores, -1e9)
            treatment_probs_full = tf.nn.softmax(treatment_scores_masked, axis=0)

            # 对 control 组计算 softmax
            control_scores_masked = tf.where(control_mask_expanded, scores, -1e9)
            control_probs_full = tf.nn.softmax(control_scores_masked, axis=0)

            # eq.(19)-(20) 合并两组的概率，得到最终的 pi
            # 一个样本的 pi 要么来自 treatment 组的 softmax，要么来自 control 组
            pi = treatment_probs_full + control_probs_full
            pi = tf.squeeze(pi, axis=-1) # 降维 [batch_size, 1] -> [batch_size]

            # --- 监控与调试: 主动错误检测 ---
            pi = tf.debugging.check_numerics(pi, "NaN/Inf in pi prediction")

            # 构造 indicator: W_i = 1 if t==1, else 0
            w_i = tf.cast(tf.equal(treatment, 1), tf.float32)  # 1 for treatment, 0 for control

            # 计算加权平均的 treatment effect
            # bar_tau_r = sum(Y_r * pi * (W_i - (1-W_i)))
            #       = sum(Y_r * pi * (2*W_i - 1))
            weight = 2 * w_i - 1  # 1 for treatment, -1 for control

            # 新增：为正样本（conversion>0 或 visit>0）添加权重，类似 demo_model_v2.py 的做法
            sample_weights_conversion = tf.where(conversion > 0, self.pos_weight, 1.0)
            sample_weights_visit = tf.where(visit > 0, self.pos_weight, 1.0)

            # eq.(21-22）
            bar_tau_r = tf.reduce_sum(conversion * pi * weight * sample_weights_conversion)
            bar_tau_c = tf.reduce_sum(visit * pi * weight * sample_weights_visit)

            # --- 修正：结合您的想法，构建一个尺度归一化且稳定的损失函数 ---
            # 您的提议 (C-R)/R 旨在解决 C-R 的尺度问题，这是一个很好的方向。
            # 但直接使用 bar_tau_r_raw 作分母会再次引入不稳定问题。
            # 我们可以结合之前的思路：用 softplus 来稳定分母。
            # 新的损失函数: (bar_tau_c_raw - bar_tau_r_raw) / (softplus(bar_tau_r_raw) + epsilon)
            # 1. 分子 (C-R) 提供了清晰的优化方向：最小化C，最大化R。
            # 2. 分母 softplus(R) 解决了尺度问题，同时保证为正，避免了数值不稳定。
            epsilon = 1e-8
            stable_denominator = tf.nn.softplus(bar_tau_r) + epsilon
            total_loss = (bar_tau_c - bar_tau_r) / stable_denominator

            # ✅ 公式 (23): loss = visit_mean / reward_mean。整个过程中，所有张量的第一个维度始终代表批次中的样本，并且顺序从未被打乱
#             total_loss = bar_tau_c / bar_tau_r
            # --- 监控与调试: 检查最终loss ---
            total_loss = tf.debugging.check_numerics(total_loss, "NaN/Inf in total_loss")

        # 反向传播
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # --- 全面监控记录到 TensorBoard ---
        step = self.optimizer.iterations

        # 思路1: 监控前向传播
        self._add_summaries("labels/conversion", conversion, step=step)
        self._add_summaries("labels/visit", visit, step=step)
        self._add_summaries("predictions/pi", pi, step=step)
        self._add_summaries("activations/shared_output", shared_output, step=step)

        # 监控 User Tower 的每一层激活
        for name, activation in user_tower_activations.items():
            self._add_summaries(f"activations/user_tower/{name}", activation, step=step)

        # 监控损失和中间计算值
        tf.summary.scalar("losses/total_loss", total_loss, step=step)
        tf.summary.scalar("calculation/bar_tau_r", bar_tau_r, step=step)
        tf.summary.scalar("calculation/bar_tau_c", bar_tau_c, step=step)
        self._add_summaries("calculation/weight", weight, step=step)

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
                if i % 10 == 0: # 每隔10个变量记录一次
                    self._add_summaries(f"gradients/var_{i}", grad, step=step)
        # --- 调试结束 ---

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "total_loss": total_loss,
            "bar_tau_r": bar_tau_r,
            "bar_tau_c": bar_tau_c,
            "ratio": bar_tau_c / bar_tau_r,
            "pi_mean": tf.reduce_mean(pi),
#             "raw_score_mean": tf.reduce_mean(raw_scores),
        }

    def test_step(self, data):
        features, labels = data

        # 前向传播
        predictions = self(features, training=False)
        scores = predictions['score']  # shape: [batch_size]

        # 计算loss需要获取真实标签
        conversion = labels['conversion']
        visit = labels['visit']
        treatment = labels['_treatment_index']  # shape: [batch_size], 1 或 0
        
        # --- 在 test_step 中进行分组 Softmax 来计算 pi ---
        treatment_mask = tf.equal(treatment, 1)
        control_mask = tf.equal(treatment, 0)
        
        treatment_mask_expanded = tf.expand_dims(treatment_mask, axis=1)
        control_mask_expanded = tf.expand_dims(control_mask, axis=1)

        treatment_scores_masked = tf.where(treatment_mask_expanded, scores, -1e9)
        treatment_probs_full = tf.nn.softmax(treatment_scores_masked, axis=0)

        control_scores_masked = tf.where(control_mask_expanded, scores, -1e9)
        control_probs_full = tf.nn.softmax(control_scores_masked, axis=0)

        pi = treatment_probs_full + control_probs_full
        pi = tf.squeeze(pi, axis=-1)

        # 构造 indicator: W_i = 1 if t==1, else 0
        w_i = tf.cast(tf.equal(treatment, 1), tf.float32)
        weight = 2 * w_i - 1

        # 新增：为正样本（conversion>0 或 visit>0）添加权重
        sample_weights_conversion = tf.where(conversion > 0, self.pos_weight, 1.0)
        sample_weights_visit = tf.where(visit > 0, self.pos_weight, 1.0)
        
        # eq.(21-22）
        bar_tau_r = tf.reduce_sum(conversion * pi * weight * sample_weights_conversion)
        bar_tau_c = tf.reduce_sum(visit * pi * weight * sample_weights_visit)

        # 计算损失
        epsilon = 1e-8
        stable_denominator = tf.nn.softplus(bar_tau_r) + epsilon
        total_loss = (bar_tau_c - bar_tau_r) / stable_denominator

        return {
            "total_loss": total_loss,
            "bar_tau_r": bar_tau_r,
            "bar_tau_c": bar_tau_c,
            "ratio": bar_tau_c / bar_tau_r,
            "pi_mean": tf.reduce_mean(pi),
        }
