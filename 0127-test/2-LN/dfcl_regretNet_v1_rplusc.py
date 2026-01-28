import tensorflow as tf
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
import os
import sys
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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

class ResidualMLPBlock(tf.keras.layers.Layer):
    """
    残差 MLP 块： (Dense -> Act -> Dropout -> Dense) + Skip，然后做 LayerNorm
    好处：可以把网络加深，但梯度更稳定，不容易“越堆越难训”
    """
    def __init__(self, hidden_dim, dropout=0.2, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.activation = activation

        self.fc1 = tf.keras.layers.Dense(
            hidden_dim, activation=None, kernel_initializer="glorot_normal"
        )
        self.act1 = tf.keras.layers.Activation(activation)
        self.dp1 = tf.keras.layers.Dropout(dropout)

        self.fc2 = tf.keras.layers.Dense(
            hidden_dim, activation=None, kernel_initializer="glorot_normal"
        )

        # 如果输入维度 != hidden_dim，用投影对齐残差
        self.proj = None

        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.act_out = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if in_dim != self.hidden_dim:
            self.proj = tf.keras.layers.Dense(
                self.hidden_dim, activation=None, kernel_initializer="glorot_normal"
            )
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.fc1(x)
        h = self.act1(h)
        h = self.dp1(h, training=training)
        h = self.fc2(h)

        skip = self.proj(x) if self.proj is not None else x
        y = self.ln(h + skip)
        y = self.act_out(y)
        return y
    

class EcomDFCL_regretNet_rplusc(tf.keras.Model):
    """
    使用 TensorFlow 2.x Keras API 实现的电商模型。
    该模型采用增广拉格朗日方法进行约束优化。
    只参考形式，但还没有完全还原。
    """
    def __init__(self, rho=0.1, dense_stats=None, fcd_mode='log1p', lambda_update_frequency=20, max_multiplier=1.0, tau=1.0, **kwargs):
        super().__init__(**kwargs)
        self.paid_pos_weight = 99.71/(100-99.71)
        self.cost_pos_weight=95.30/(100-95.30)
        
        # --- 增广拉格朗日方法超参数 ---
        self.rho = rho  # 二次惩罚项的系数 ρ
        self.lambda_update_frequency = lambda_update_frequency # 拉格朗日乘子 λ 的更新频率 Q
        self.max_multiplier = max_multiplier
        self.tau = tau
        self.mu = tf.Variable(0.0, trainable=False, name='lagrange_multiplier_mu')
        
        # 从 fsfc.py 导入配置
        self.sparse_feature_names = SPARSE_FEATURE_NAME
        self.dense_feature_names = DENSE_FEATURE_NAME
        self.sparse_feature_slot_map = SPARSE_FEATURE_NAME_SLOT_ID
        self.num_estimated_vec_features = 10000# 为什么要这么大数量 12000000

        # ===== 新增 =====
        # dense_stats 期望格式：
        # {
        #   "raw":   {"mean": [..], "std": [..]},
        #   "log1p": {"mean": [..], "std": [..]}
        # }
        self.fcd_mode = fcd_mode
        self._dense_global_mean = None  # shape: [num_dense]
        self._dense_global_std = None   # shape: [num_dense]
        if dense_stats is not None:
            stats_obj = dense_stats.get(self.fcd_mode, dense_stats)
            mean_list = stats_obj.get("mean")
            std_list = stats_obj.get("std")
            if mean_list is None or std_list is None:
                raise ValueError(f"dense_stats 缺少 mean/std: keys={list(stats_obj.keys())}")

            mean = tf.convert_to_tensor(mean_list, dtype=tf.float32)
            std = tf.convert_to_tensor(std_list, dtype=tf.float32)

            self._dense_global_mean = mean
            self._dense_global_std = std
        # ===== 新增结束 =====

        # 模型超参数 TODO：需要依据实际数据集进行修改
        self.sparse_feature_dim = 8 # TODO：简化为4
        self.dense_feature_dim = 1
        self.treatment_order = [1, 0] #处理组为15off，另一组是空白组
        self.ratios = [i / 100.0 for i in range(5, 105, 5)] #ratio也就是lambda，这里应该换成更为密集的，真正模拟积分。
        self.targets = ['paid', 'cost']
        
        self.total_samples = statistical_config['N']
        self.treatment_sample_counts = {
            1: statistical_config['N1'],
            0: statistical_config['N0']
        }
        
        # 特征处理层
        self._build_feature_layers()
        
        # 网络结构，user_tower为共享底层
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'), # 512->128
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),# 256->64
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LayerNormalization(),
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
        # ======= 修改开始 =======
        # for feature_name in self.dense_feature_names:
        #     fcd = inputs[feature_name]
        #     fcd = tf.math.log1p(tf.maximum(fcd,0.0))
        #     fcd = (fcd - tf.reduce_mean(fcd)) / (tf.math.reduce_std(fcd) + 1e-8)
        #     dense_vectors.append(tf.reshape(fcd, [-1, self.dense_feature_dim]))
        for i, feature_name in enumerate(self.dense_feature_names):
            # fcd变换之前记录数据分布
            raw_val = inputs[feature_name]
            if training:
                if not hasattr(self, "_last_raw_inputs"):
                    self._last_raw_inputs = {}
                # 保存原始分布 (展平以便计算统计量)
                self._last_raw_inputs[feature_name] = tf.reshape(raw_val, [-1])

            fcd = tf.cast(inputs[feature_name], tf.float32)
            fcd = tf.maximum(fcd, 0.0)
            if self.fcd_mode == 'log1p':
                fcd = tf.math.log1p(fcd)
            
            if self._dense_global_mean is not None and self._dense_global_std is not None:
                mean = self._dense_global_mean[i]
                std = self._dense_global_std[i]
                fcd = (fcd - mean) / (std + 1e-8)
            else:
                fcd = (fcd - tf.reduce_mean(fcd)) / (tf.math.reduce_std(fcd) + 1e-8)

            dense_vectors.append(tf.reshape(fcd, [-1, self.dense_feature_dim]))
            
            # fcd变换之后记录数据分布
            if training:
                if not hasattr(self, "_last_fcd"):
                    self._last_fcd = {}
                self._last_fcd[feature_name] = tf.reshape(fcd, [-1])  # 展平成一维便于 histogram
        # ======= 修改结束 =======

        concat_input = tf.concat(sparse_vectors + dense_vectors, axis=1)
        
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

    def compute_local_losses(self, predictions, labels):
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
            # loss_tensor = tf.constant(0.0, dtype=tf.float32)
            for treatment in self.treatment_order:
                pred_name = f"{target_name}_treatment_{treatment}"
                logit = predictions[pred_name]

                out = tf.minimum(logit, 10.0)
                label = tf.cast(labels[target_name], tf.float32) 

                term1 = -label * out
                term2 = (1 + label) * (tf.maximum(out, 0) + tf.math.log(1 + tf.exp(-tf.abs(out))))
                loss_per_sample = term1 + term2

                treatment_mask = tf.cast(tf.equal(treatment_idx, treatment), tf.float32)

                sample_weights = tf.where(label > 0, pos_weight, 1.0)
                weighted_loss_per_sample = loss_per_sample * sample_weights
                masked_loss = weighted_loss_per_sample * treatment_mask
                local_loss += tf.reduce_mean(masked_loss)

            if target_name == 'paid':
                paid_loss += local_loss
            else:
                cost_loss += local_loss

        return paid_loss, cost_loss

    def decision_learning_objective_term(self, predictions, labels): # 5.3 引入温度参数τ和最大熵正则化，使得决策损失更加平滑且可微
        decision_loss_sum = tf.constant(0.0, dtype=tf.float32)
        # 预先计算 treatment_mask（仅一次）
        treatment_idx = tf.cast(labels['treatment'], tf.int32)

        # pred_dict = {key: tf.exp(tf.minimum(logit, 10.0)) for key, logit in predictions.items()}
        # pred_dict = {key: tf.minimum(logit, 10.0) for key, logit in predictions.items()}
        pred_dict = {key: tf.minimum(tf.nn.sigmoid(logit), 10.0) for key, logit in predictions.items()}

        for ratio in self.ratios: # 模拟决策过程
            cancat_list, mask_list = [], []
            for treatment in self.treatment_order:
                v = pred_dict["paid_treatment_{}".format(treatment)] - ratio * pred_dict["cost_treatment_{}".format(treatment)]
                cancat_list.append(tf.reshape(v, [-1, 1]))
                
                treatment_mask = tf.equal(treatment, treatment_idx)
                mask_list.append(tf.reshape(tf.cast(treatment_mask, tf.float32), [-1, 1]))

            # 应用温度调节的softmax
            cancat_tensor = tf.concat(cancat_list, axis=1)
            softmax_tensor = tf.nn.softmax(cancat_tensor / self.tau, axis=1)
            
            mask_tensor = tf.concat(mask_list, axis=1) # 样本真实 treatment 的 one-hot 编码
            ratio_target = tf.reshape(labels['paid'] - ratio * labels['cost'], [-1, 1]) #样本的真实收益
            # cancat_tensor * mask_tensor 的结果是，只保留模型对真实 treatment 的“最优概率”预测，其他位置为0
            decision_loss = tf.reduce_mean(softmax_tensor * mask_tensor * ratio_target)

            decision_loss_sum += decision_loss
            
        # return decision_loss_sum / len(self.ratios)
        return decision_loss_sum


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
    
    def _compute_grad_dot_product(self, grads1, grads2):
        """计算两组梯度的点积（grad1 · grad2），用于判断任务梯度是否冲突"""
        dot_product = tf.constant(0.0, dtype=tf.float32)
        for g1, g2 in zip(grads1, grads2):
            if g1 is not None and g2 is not None:
                dot_product += tf.reduce_sum(g1 * g2)
        return dot_product


    def train_step(self, data):
        """
        使用增广拉格朗日方法实现约束优化。
        - 主目标：最大化 decision_loss (即最小化 -decision_loss)
        - 约束：paid_loss 和 cost_loss 趋近于0
        """
        features, labels = data
        step = self.optimizer.iterations

        with tf.GradientTape(persistent=True) as tape:
            # 1. 前向传播，计算所有损失
            predictions = self(features, training=True)

            # 从模型属性中获取中间激活值
            shared_output = self._last_shared_output
            user_tower_activations = self._last_user_tower_activations

            decision_loss = self.decision_learning_objective_term(predictions, labels)  # maximize
            paid_loss, cost_loss = self.compute_local_losses(predictions, labels)

            # 检查数值稳定性
            paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in paid_loss")
            cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in cost_loss")
            decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in decision_loss")

            # 2. 构建增广拉格朗日函数
            # L(w, μ) = f(w) + μ * g(w)^2 + (ρ/2) * g(w)^2
            # f(w) 是主目标: -decision_loss (因为我们要最大化 decision_loss)
            # g(w) 是约束函数: prediction_loss = paid_loss + cost_loss (我们希望它为0)
            # μ 是拉格朗日乘子: self.mu
            prediction_loss = paid_loss + cost_loss

            # 拉格朗日项: μ * g(w)
            # 在更新w时，μ被视为常数，因此停止梯度回传
            lambda_term = tf.stop_gradient(self.mu) * prediction_loss

            # 二次惩罚项: (ρ/2) * g(w)
            penalty_term = (self.rho / 2.0) * prediction_loss

            # 最终用于更新模型参数 w 的总损失
            model_update_loss = -decision_loss + lambda_term + penalty_term


        # 3. 计算梯度并更新模型参数 w (Primal Update)
        model_variables = self.trainable_variables
        model_gradients = tape.gradient(model_update_loss, model_variables)
        # 单独计算 task_loss 的梯度，用于分析
        prediction_loss_gradients = tape.gradient(prediction_loss, model_variables)
        # 单独计算 decision_loss 的梯度，用于分析
        decision_loss_gradients = tape.gradient(decision_loss, model_variables)
        
        # ===== 新增：分别计算 paid_loss / cost_loss 的梯度，用于任务冲突诊断 =====
        paid_loss_gradients = tape.gradient(paid_loss, model_variables)
        cost_loss_gradients = tape.gradient(cost_loss, model_variables)

        self.optimizer.apply_gradients(zip(model_gradients, model_variables))

        # 4. 每隔 Q 步，更新拉格朗日乘子 μ (Dual Update)
        def update_mu():
            # 更新规则: μ_new = μ_old + ρ * g(w)
            # 使用 tf.stop_gradient 确保此更新操作不会影响 w 的梯度计算
            # self.mu.assign_add(self.rho * tf.stop_gradient(prediction_loss))
            new_mu = self.mu + self.rho * tf.stop_gradient(prediction_loss)
            # 将 mu 限制在 [0, 1] 范围内
            self.mu.assign(tf.clip_by_value(new_mu, clip_value_min=0.0, clip_value_max=self.max_multiplier))
            return tf.constant(True)

        def no_mu_update():
            return tf.constant(False)
        
        # 使用 tf.cond 实现条件更新
        tf.cond(
            tf.equal(step % self.lambda_update_frequency, 0),
            update_mu,
            no_mu_update
        )

        # --- 梯度分析与监控 ---
        step = self.optimizer.iterations

        # 把每个feature的 fcd 变换之前的分布写进 TensorBoard
        if hasattr(self, "_last_raw_inputs"):
            for fname, ftensor in self._last_raw_inputs.items():
                # 存放在 preprocess/raw/ 下，与 preprocess/fcd/ 形成对比
                self._add_summaries(f"preprocess/raw/{fname}", ftensor, step=step)

        # 把每个feature的 fcd 变换之后的分布写进 TensorBoard
        if hasattr(self, "_last_fcd"):
            for fname, ftensor in self._last_fcd.items():
                self._add_summaries(f"preprocess/fcd/{fname}", ftensor, step=step)
        
        # 计算并记录各部分梯度的范数（大小）
        valid_task_grads = [g for g in prediction_loss_gradients if g is not None]
        if valid_task_grads:
            task_grad_norm = tf.linalg.global_norm(valid_task_grads)
            tf.summary.scalar("gradients_analysis/norm_prediction_loss", task_grad_norm, step=step)

        valid_paid_grads = [g for g in paid_loss_gradients if g is not None]
        if valid_paid_grads:
            paid_grad_norm = tf.linalg.global_norm(valid_paid_grads)
            tf.summary.scalar("gradients_analysis/norm_paid_loss", paid_grad_norm, step=step)
        
        valid_cost_grads = [g for g in cost_loss_gradients if g is not None]
        if valid_cost_grads:
            cost_grad_norm = tf.linalg.global_norm(valid_cost_grads)
            tf.summary.scalar("gradients_analysis/norm_cost_loss", cost_grad_norm, step=step)

        valid_decision_grads = [g for g in decision_loss_gradients if g is not None]
        if valid_decision_grads:
            decision_grad_norm = tf.linalg.global_norm(valid_decision_grads)
            tf.summary.scalar("gradients_analysis/norm_decision_loss", decision_grad_norm, step=step)
        
        cosine_similarity = self._compute_cosine_similarity(prediction_loss_gradients, decision_loss_gradients)
        tf.summary.scalar("gradients_analysis/cosine_similarity_task_vs_decision", cosine_similarity, step=step)
        
        # ===== 新增：paid vs cost 梯度内积（dot product）判断是否冲突 =====
        grad_dot_paid_cost = self._compute_grad_dot_product(paid_loss_gradients, cost_loss_gradients)
        tf.summary.scalar("gradients_analysis/dot_paid_vs_cost", grad_dot_paid_cost, step=step)
        # tf.print("Gradient dot product (paid vs cost):", grad_dot_paid_cost)

        # 监控前向传播
        self._add_summaries("labels/paid", labels['paid'], step=step)
        self._add_summaries("labels/cost", labels['cost'], step=step)
        self._add_summaries("activations/shared_output",  shared_output, step=step)
        # 5. 记录监控指标到 TensorBoard
        self._add_summaries("losses/1_paid_loss", paid_loss, step=step)
        self._add_summaries("losses/2_cost_loss", cost_loss, step=step)
        self._add_summaries("losses/3_decision_loss", decision_loss, step=step)
        self._add_summaries("losses/4_model_update_loss", model_update_loss, step=step)
        self._add_summaries("losses/5_prediction_loss", prediction_loss, step=step)
        tf.summary.scalar("lagrangian/mu", self.mu, step=step)
        tf.summary.scalar("lagrangian/lambda_term", lambda_term, step=step)
        tf.summary.scalar("lagrangian/penalty_term", penalty_term, step=step)

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
        
        # 监控梯度
        # 过滤掉None的梯度
        valid_gradients = [g for g in model_gradients if g is not None]
        if valid_gradients:
            global_norm = tf.linalg.global_norm(valid_gradients)
            tf.summary.scalar("gradients/global_norm", global_norm, step=step)
            # 抽样监控几个关键层的梯度
            for i, grad in enumerate(valid_gradients):
                if i % 10 == 0: # 每隔10个变量记录一次，避免日志过大
                    self._add_summaries(f"gradients/var_{i}", grad, step=step)
        
        # 手动删除 tape 释放资源
        del tape

        # 返回用于 Keras 进度条显示的指标
        return {
            "total_loss": model_update_loss,
            "decision_loss": decision_loss, # 越大越好
            "paid_loss": paid_loss,
            "cost_loss": cost_loss,
            "lambda_term": lambda_term,
            "penalty_term": penalty_term,
            "lagrangian": self.mu,
        }

    def test_step(self, data):
        """验证步骤，不更新权重，只计算损失"""
        features, labels = data
        predictions = self(features, training=False)

        paid_loss, cost_loss = self.compute_local_losses(predictions, labels)
        decision_loss = self.decision_learning_objective_term(predictions, labels)  # maximize
        
        # 在验证集上，我们通常只关心原始损失，不计算增广拉格朗日损失
        model_update_loss = -decision_loss # 可以只看主目标

        return {
            "total_loss": model_update_loss,
            "decision_loss": decision_loss,
            "paid_loss": paid_loss, 
            "cost_loss": cost_loss,
        }