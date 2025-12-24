import tensorflow as tf
import cvxpy as cp
import numpy as np

#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from data_utils import *
# 1120修改，将treatment纳入特征，形成标准的预测网络
SPARSE_FEATURE_NAME = ["treatment"]
DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
SPARSE_FEATURE_NAME_SLOT_ID = {
    'treatment': 1024  # 为 treatment 分配一个唯一的 slot_id
}
statistical_config={
    'N':11183673,
    'N1':9506123,
    'N0':1677550
}


class RobustnessSDP:
    """使用SDP验证神经网络层的Lipschitz常数上界"""
    
    @staticmethod
    def verify_layer_lipschitz(W):
        # W: numpy array, shape [n_in, n_out] 或 [n_out, n_in] 均可
        s = np.linalg.svd(W, compute_uv=False)
        return float(s[0])  # 最大奇异值 = 谱范数


    # # @staticmethod
    # def verify_layer_lipschitz(W, activation='relu'):
    #     """
    #     验证单层的Lipschitz常数
    #     W: 权重矩阵 (output_dim, input_dim)
    #     返回:  Lipschitz常数的上界
    #     """
    #     n_in, n_out = W.shape[1], W.shape[0]
        
    #     # 创建SDP变量
    #     P = cp.Variable((n_in, n_in), PSD=True)  # 半正定矩阵
    #     gamma = cp.Variable(nonneg=True)  # Lipschitz常数
        
    #     if activation == 'relu':
    #         # ReLU激活的Lipschitz约束
    #         # 构造Schur complement约束
    #         constraints = [
    #             cp.bmat([
    #                 [P, P @ W. T],
    #                 [W @ P, gamma * np.eye(n_out)]
    #             ]) >> 0,  # 半正定约束
    #             P >> np.eye(n_in) * 1e-6  # P必须是正定的
    #         ]
    #     else:
    #         # 线性层的简化约束
    #         constraints = [W. T @ W << gamma**2 * np.eye(n_in)]
        
    #     # 最小化Lipschitz常数
    #     objective = cp.Minimize(gamma)
    #     prob = cp.Problem(objective, constraints)
    #     prob.solve(solver=cp.SCS)
        
    #     if gamma.value is None:
    #         raise RuntimeError("SDP 求解失败：gamma 未返回有效值。")
    #     return gamma.value

    @staticmethod
    def verify_decision_robustness(model, sample_features, epsilon=0.1):
        """
        验证决策对输入扰动的鲁棒性
        使用SDP检查在epsilon扰动下决策是否改变
        """
        # 获取原始预测
        original_preds = model(sample_features, training=False)
        
        # 提取权重矩阵（假设是Dense层）
        dense_layer = model.user_tower.layers[0]
        W = dense_layer.kernel.numpy()
        
        # 计算Lipschitz常数
        L = RobustnessSDP.verify_layer_lipschitz(W)
        
        # 计算决策边界的安全半径
        # 如果 |a_i - a_j| > 2*L*epsilon, 则决策在epsilon扰动下不会改变
        a_t1 = (original_preds['paid_treatment_1'] - 
                original_preds['cost_treatment_1']).numpy()
        a_t0 = (original_preds['paid_treatment_0'] - 
                original_preds['cost_treatment_0']).numpy()
        
        margin = np.abs(a_t1 - a_t0)
        safe_radius = margin / (2 * L)
        
        is_robust = safe_radius > epsilon
        
        return {
            'lipschitz_constant': L,
            'decision_margin': margin,
            'safe_radius': safe_radius,
            'is_robust': is_robust,
            'avg_margin': np.mean(margin), # 新增
            'min_safe_radius': np.min(safe_radius), # 新增
            'avg_safe_radius': np.mean(safe_radius), # 新增
            'robustness_ratio': np.mean(is_robust)
        }

class EcomDFCL_v3(tf.keras.Model): # std+ifdl一系列clip、maximum+2pos
    """
    使用 TensorFlow 2.x Keras API 实现的电商模型。
    该模型集成了 GradNorm 和自定义决策损失。
    """
    def __init__(self, alpha=1.2, **kwargs):
        super().__init__(**kwargs)
        self.paid_pos_weight = 99.71/(100-99.71)
        self.cost_pos_weight= 95.30/(100-95.30)
        self.alpha = alpha #prediction loss前面的系数
        
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
        # 【修改2】为 task_towers 的构建也明确输入维度
        # 这是第二步处理的关键：塔的输入 = user_tower输出 + treatment嵌入向量
        user_tower_output_dim = 128 # user_tower 最后一层的维度
        task_tower_input_dim = user_tower_output_dim + self.sparse_feature_dim

        for target in self.targets:
            for treatment in self.treatment_order:
                name = "{}_treatment_{}_tower".format(target, treatment)
                tower = tf.keras.Sequential([
                    tf.keras.layers.Dense(dims, activation='relu', kernel_initializer='glorot_normal') for dims in tower_dims[:-1]
                ] + [
                    tf.keras.layers.Dense(tower_dims[-1], kernel_initializer='glorot_normal')
                ], name=name)
                # 显式构建塔
                tower.build(input_shape=(None, task_tower_input_dim))
                self.task_towers[name] = tower

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
        # 新增：处理稀疏特征 (包括 treatment)
        for feature_name in self.sparse_feature_names:
            feature_input = inputs[feature_name]
            # Hashing层需要字符串输入，如果treatment是int，则转换
            if feature_input.dtype != tf.string:
                feature_input = tf.strings.as_string(feature_input)
            
            # 1. Hashing
            hashed_output = self.hashing_layers[feature_name](feature_input)
            
            # 2. Embedding
            slot_id = str(self.sparse_feature_slot_map[feature_name])
            embedding_vector = self.embedding_layers[slot_id](hashed_output)
            sparse_vectors.append(embedding_vector)

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
        # 根据 __init__ 中的定义，塔的输入是 shared_output 和 treatment 嵌入的拼接
        # sparse_vectors[0] 是 treatment 嵌入，因为它是唯一的稀疏特征
        treatment_embedding = sparse_vectors[0]
        tower_input = tf.concat([shared_output, treatment_embedding], axis=1)
        for name, tower in self.task_towers.items():
            # name is like "paid_treatment_30_tower"
            pred_name = name.replace('_tower', '')
            logit = tower(tower_input, training=training)
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
                pred_dict[f"paid_treatment_{t}"] - ratio * pred_dict[f"cost_treatment_{t}"]
                for t in self.treatment_order
            ]
            cancat_tensor = tf.nn.softmax(tf.stack(values, axis=1), axis=1)

            # mask_tensor 是 one-hot 编码
            mask_tensor = tf.stack([treatment_masks[t] for t in self.treatment_order], axis=1)

            ratio_target = tf.reshape(labels['paid'] - ratio * labels['cost'], [-1, 1])#样本的真实收益
            # cancat_tensor * mask_tensor 的结果是，只保留模型对真实 treatment 的“最优概率”预测，其他位置为0
            decision_loss = tf.reduce_sum(cancat_tensor * mask_tensor * ratio_target * weight_tensor)

            decision_loss_sum += decision_loss

        return decision_loss_sum / len(self.ratios)
    
    def decision_entropy_regularized_loss(self, predictions, labels): # 5.3 引入温度参数τ和最大熵正则化，使得决策损失更加平滑且可微
        decision_loss_sum = tf.constant(0.0, dtype=tf.float32)
        # 预先计算 treatment_mask（仅一次）
        treatment_idx = tf.cast(labels['treatment'], tf.int32)
        treatment_masks = {
            t: tf.cast(tf.equal(treatment_idx, t), tf.float32) for t in self.treatment_order
        }
        
        pred_dict = {key: tf.exp(tf.minimum(logit, 10.0)) for key, logit in predictions.items()}
        
        # 温度参数τ - 可设置为超参数，softmax更加平滑（τ>1）或更加尖锐（τ<1）。
        tau = self.tau if hasattr(self, 'tau') else 1.2
        
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

        for ratio in self.ratios: # 模拟决策过程
            cancat_list, mask_list = [], []
            for treatment in self.treatment_order:
                v = pred_dict["paid_treatment_{}".format(treatment)] - ratio * pred_dict["cost_treatment_{}".format(treatment)]
                cancat_list.append(tf.reshape(v, [-1, 1]))
                
                treatment_idx = tf.cast(labels['treatment'], tf.int32)
                treatment_mask = tf.equal(treatment, treatment_idx)
                mask_list.append(tf.reshape(tf.cast(treatment_mask, tf.float32), [-1, 1]))

            # 应用温度调节的softmax
            cancat_tensor = tf.concat(cancat_list, axis=1)
            softmax_tensor = tf.nn.softmax(cancat_tensor / tau, axis=1)
            
            mask_tensor = tf.concat(mask_list, axis=1) # 样本真实 treatment 的 one-hot 编码
            ratio_target = tf.reshape(labels['paid'] - ratio * labels['cost'], [-1, 1]) #样本的真实收益
            # cancat_tensor * mask_tensor 的结果是，只保留模型对真实 treatment 的“最优概率”预测，其他位置为0
            decision_loss = tf.reduce_sum(softmax_tensor * mask_tensor * ratio_target* weight_tensor)
            decision_loss_sum += decision_loss
            
        return decision_loss_sum / len(self.ratios)

    
    # 更加合理的5.4复现
    def decision_improved_finite_difference_loss(self, predictions, labels):
        """
        修复后的改进有限差分策略损失函数
        主要修复：
        1. 梯度计算的分母添加安全检查
        2. 修正概率计算方式，使用batch内统计而非预设值
        3. 限制梯度幅度，防止数值爆炸
        4. 添加数值稳定性保护
        """
        # 数值稳定性参数
        epsilon = 1e-6  # 增大epsilon值
        max_grad_magnitude = 10.0  # 限制梯度最大幅度
        
        # 初始化总梯度累加器
        total_paid_grads = {f"paid_treatment_{t}": tf.zeros_like(predictions[f"paid_treatment_{t}"])
                        for t in self.treatment_order}
        total_cost_grads = {f"cost_treatment_{t}": tf.zeros_like(predictions[f"cost_treatment_{t}"])
                            for t in self.treatment_order}

        # 获取样本数和treatment数
        batch_size = tf.shape(labels['treatment'])[0]
        num_treatments = len(self.treatment_order)

        # 【修复1】：使用batch内的实际分布计算概率，而非预设统计值
        treatment_idx = tf.cast(labels['treatment'], tf.int32)
        N = tf.cast(batch_size, tf.float32)
        p_t_dict = {}
        for t in self.treatment_order:
            N_t = tf.reduce_sum(tf.cast(tf.equal(treatment_idx, t), tf.float32))
            # 使用更大的平滑项，避免除以接近0的数
            p_t_dict[t] = tf.maximum(N_t / N, 0.01)  # 至少1%的概率

        # 对每个λ(ratio)计算梯度
        for ratio in self.ratios:
            # 计算每个样本每个treatment的得分 a_ij = r_hat_ij - λ * c_hat_ij
            a_values = {t: predictions[f"paid_treatment_{t}"] - ratio * predictions[f"cost_treatment_{t}"]
                        for t in self.treatment_order}

            # 堆叠所有treatment的得分 [batch_size, num_treatments]
            a_matrix = tf.stack(list(a_values.values()), axis=1)

            # 找到每个样本的最优treatment (argmax_j a_ij)
            optimal_treatment = tf.argmax(a_matrix, axis=1, output_type=tf.int32)

            # 计算每个样本的实际treatment是否与最优treatment匹配
            actual_treatment_onehot = tf.one_hot(treatment_idx, depth=num_treatments)
            optimal_mask_onehot = tf.one_hot(optimal_treatment, depth=num_treatments)
            is_matching = tf.reduce_sum(optimal_mask_onehot * actual_treatment_onehot, axis=1)

            # 分离匹配和不匹配的样本索引
            matching_indices = tf.where(is_matching > 0.5)
            mismatching_indices = tf.where(is_matching < 0.5)

            # --- 为 tf.cond 定义分支函数 ---
            def compute_deltas(indices, is_matching_case):
                """为匹配或不匹配的样本计算梯度增量"""
                paid_deltas = {k: tf.zeros_like(v) for k, v in total_paid_grads.items()}
                cost_deltas = {k: tf.zeros_like(v) for k, v in total_cost_grads.items()}

                indices = tf.squeeze(indices, axis=1)
                actual_t = tf.gather(treatment_idx, indices)
                a_sub_matrix = tf.gather(a_matrix, indices)
                paid_actual = tf.gather(labels['paid'], indices)
                cost_actual = tf.gather(labels['cost'], indices)
                
                p_t_gathered = tf.gather([p_t_dict[t] for t in self.treatment_order], actual_t)
                
                # 【修复2】：限制基础梯度分量的幅度
                base_grad_component = (paid_actual - ratio * cost_actual) / (N * p_t_gathered)
                base_grad_component = tf.clip_by_value(base_grad_component, -max_grad_magnitude, max_grad_magnitude)

                if is_matching_case:
                    grad_component = -base_grad_component
                    a_actual_vals = tf.reduce_sum(a_sub_matrix * tf.one_hot(actual_t, depth=num_treatments), axis=1)
                    mask_actual = tf.one_hot(actual_t, depth=num_treatments, on_value=-1e10, off_value=0.0)
                    max_a_without_actual = tf.reduce_max(a_sub_matrix + mask_actual, axis=1)
                    
                    # 【修复3】：为分母项添加安全边界
                    h_r_actual = tf.maximum(max_a_without_actual - a_actual_vals, epsilon)
                    h_c_actual = tf.maximum(tf.abs((a_actual_vals - max_a_without_actual) / ratio), epsilon)

                    for t in self.treatment_order:
                        is_actual_t_mask = tf.equal(actual_t, t)
                        
                        # Case 1: t is the actual treatment
                        update_r = tf.clip_by_value(grad_component / h_r_actual, -max_grad_magnitude, max_grad_magnitude)
                        update_c = tf.clip_by_value(grad_component / h_c_actual, -max_grad_magnitude, max_grad_magnitude)
                        
                        # Case 2: t is not the actual treatment
                        a_t = tf.gather(a_values[t], indices)
                        h_r_other = tf.maximum(tf.abs(a_actual_vals - a_t), epsilon)
                        h_c_other = tf.maximum(tf.abs((a_t - a_actual_vals) / ratio), epsilon)
                        update_r_other = tf.clip_by_value(grad_component / h_r_other, -max_grad_magnitude, max_grad_magnitude)
                        update_c_other = tf.clip_by_value(grad_component / h_c_other, -max_grad_magnitude, max_grad_magnitude)

                        total_update_r = tf.where(is_actual_t_mask, update_r, update_r_other)
                        total_update_c = tf.where(is_actual_t_mask, update_c, update_c_other)

                        paid_deltas[f"paid_treatment_{t}"] = tf.tensor_scatter_nd_add(
                            paid_deltas[f"paid_treatment_{t}"], 
                            tf.expand_dims(indices, 1), 
                            total_update_r
                        )
                        cost_deltas[f"cost_treatment_{t}"] = tf.tensor_scatter_nd_add(
                            cost_deltas[f"cost_treatment_{t}"], 
                            tf.expand_dims(indices, 1), 
                            total_update_c
                        )
                else: # Mismatching case
                    grad_component = base_grad_component
                    optimal_t = tf.gather(optimal_treatment, indices)
                    a_actual_vals = tf.reduce_sum(a_sub_matrix * tf.one_hot(actual_t, depth=num_treatments), axis=1)
                    a_optimal_vals = tf.reduce_sum(a_sub_matrix * tf.one_hot(optimal_t, depth=num_treatments), axis=1)

                    # 【修复4】：确保分母项为正值且有安全边界
                    h_r_actual = tf.maximum(tf.abs(a_optimal_vals - a_actual_vals), epsilon)
                    h_c_actual = tf.maximum(tf.abs((tf.reduce_max(a_sub_matrix, axis=1) - a_actual_vals) / ratio), epsilon)
                    h_r_optimal = h_r_actual  # 对称性
                    h_c_optimal = h_c_actual

                    for t in self.treatment_order:
                        update_r = tf.zeros_like(actual_t, dtype=tf.float32)
                        update_c = tf.zeros_like(actual_t, dtype=tf.float32)

                        # Add contribution from actual treatment
                        actual_contrib_r = tf.where(
                            tf.equal(actual_t, t), 
                            tf.clip_by_value(grad_component / h_r_actual, -max_grad_magnitude, max_grad_magnitude), 
                            0.0
                        )
                        actual_contrib_c = tf.where(
                            tf.equal(actual_t, t), 
                            tf.clip_by_value(grad_component / h_c_actual, -max_grad_magnitude, max_grad_magnitude), 
                            0.0
                        )
                        
                        # Add contribution from optimal treatment
                        optimal_contrib_r = tf.where(
                            tf.equal(optimal_t, t), 
                            tf.clip_by_value(grad_component / h_r_optimal, -max_grad_magnitude, max_grad_magnitude), 
                            0.0
                        )
                        optimal_contrib_c = tf.where(
                            tf.equal(optimal_t, t), 
                            tf.clip_by_value(grad_component / h_c_optimal, -max_grad_magnitude, max_grad_magnitude), 
                            0.0
                        )
                        
                        update_r = actual_contrib_r + optimal_contrib_r
                        update_c = actual_contrib_c + optimal_contrib_c
                        
                        paid_deltas[f"paid_treatment_{t}"] = tf.tensor_scatter_nd_add(
                            paid_deltas[f"paid_treatment_{t}"], 
                            tf.expand_dims(indices, 1), 
                            update_r
                        )
                        cost_deltas[f"cost_treatment_{t}"] = tf.tensor_scatter_nd_add(
                            cost_deltas[f"cost_treatment_{t}"], 
                            tf.expand_dims(indices, 1), 
                            update_c
                        )

                return list(paid_deltas.values()) + list(cost_deltas.values())

            def return_zero_deltas():
                """返回一个结构正确的零值列表"""
                return [tf.zeros_like(predictions[f"paid_treatment_{t}"]) for t in self.treatment_order] + \
                    [tf.zeros_like(predictions[f"cost_treatment_{t}"]) for t in self.treatment_order]

            # --- 使用 tf.cond 独立计算增量 ---
            matching_deltas_list = tf.cond(
                tf.size(matching_indices) > 0,
                lambda: compute_deltas(matching_indices, is_matching_case=True),
                return_zero_deltas
            )
            mismatching_deltas_list = tf.cond(
                tf.size(mismatching_indices) > 0,
                lambda: compute_deltas(mismatching_indices, is_matching_case=False),
                return_zero_deltas
            )

            # --- 在循环中累加当前 ratio 计算出的梯度 ---
            for i, t in enumerate(self.treatment_order):
                paid_key = f"paid_treatment_{t}"
                cost_key = f"cost_treatment_{t}"
                total_paid_grads[paid_key] += matching_deltas_list[i] + mismatching_deltas_list[i]
                total_cost_grads[cost_key] += matching_deltas_list[i + num_treatments] + mismatching_deltas_list[i + num_treatments]

        # 【修复5】：对最终梯度进行额外的数值检查和裁剪
        for key in total_paid_grads:
            total_paid_grads[key] = tf.clip_by_value(
                total_paid_grads[key], 
                -max_grad_magnitude * len(self.ratios), 
                max_grad_magnitude * len(self.ratios)
            )
            total_paid_grads[key] = tf.where(
                tf.math.is_finite(total_paid_grads[key]), 
                total_paid_grads[key], 
                tf.zeros_like(total_paid_grads[key])
            )
        
        for key in total_cost_grads:
            total_cost_grads[key] = tf.clip_by_value(
                total_cost_grads[key], 
                -max_grad_magnitude * len(self.ratios), 
                max_grad_magnitude * len(self.ratios)
            )
            total_cost_grads[key] = tf.where(
                tf.math.is_finite(total_cost_grads[key]), 
                total_cost_grads[key], 
                tf.zeros_like(total_cost_grads[key])
            )

        # 构建有限差分损失（梯度与预测值的点积）
        ifdl_loss = tf.constant(0.0, dtype=tf.float32)
        for t in self.treatment_order:
            paid_key = f"paid_treatment_{t}"
            cost_key = f"cost_treatment_{t}"
            
            # 【修复6】：对点积结果也进行数值检查
            conv_contrib = tf.reduce_sum(total_paid_grads[paid_key] * predictions[paid_key])
            cost_contrib = tf.reduce_sum(total_cost_grads[cost_key] * predictions[cost_key])
            
            conv_contrib = tf.where(tf.math.is_finite(conv_contrib), conv_contrib, 0.0)
            cost_contrib = tf.where(tf.math.is_finite(cost_contrib), cost_contrib, 0.0)
            
            ifdl_loss += conv_contrib + cost_contrib

        # 【修复7】：对最终损失进行范围限制
        ifdl_loss = tf.clip_by_value(ifdl_loss, -1e6, 1e6)
        
        return ifdl_loss
    
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
                # 【修复】将 IndexedSlices 转换为密集张量以支持乘法操作
                # 梯度可能是 IndexedSlices 类型（例如来自Embedding层），需要转换为密集张量进行计算
                g1_dense = tf.convert_to_tensor(g1)
                g2_dense = tf.convert_to_tensor(g2)
                dot_product += tf.reduce_sum(g1_dense * g2_dense)
                norm1_sq += tf.reduce_sum(g1_dense * g1_dense)
                norm2_sq += tf.reduce_sum(g2_dense * g2_dense)
        
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

            paid_loss, cost_loss = self.compute_local_losses(predictions, labels)
            decision_loss = self.decision_policy_learning_loss(predictions, labels)
            
            # --- 进阶监控 3 (续): 检查最终loss ---
            paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in paid_loss")
            cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in cost_loss")
            decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in decision_loss")
            
            # 2. 计算用于更新【模型参数】的损失
            weighted_task_loss = 0.5 * paid_loss + 0.5 * cost_loss
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
        self._add_summaries("labels/paid", labels['paid'], step=step)
        self._add_summaries("labels/cost", labels['cost'], step=step)
        self._add_summaries("activations/shared_output",  shared_output, step=step)

        # 监控损失分量
        self._add_summaries("losses/1_paid_loss", paid_loss, step=step)
        self._add_summaries("losses/2_cost_loss", cost_loss, step=step)
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
            "paid_loss": paid_loss, "cost_loss": cost_loss,
        }

    def test_step(self, data):
        """验证步骤，不更新权重，只计算损失"""
        features, labels = data
        predictions = self(features, training=False)
        
        for name, pred in predictions.items():
            predictions[name] = tf.debugging.check_numerics(pred, f"NaN/Inf in validation prediction: {name}")

        paid_loss, cost_loss = self.compute_local_losses(predictions, labels)
        decision_loss = self.decision_policy_learning_loss(predictions, labels)
        
        paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in val_paid_loss")
        cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in val_cost_loss")
        decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in val_decision_loss")

        weighted_task_loss = 0.5 * paid_loss + 0.5 * cost_loss
        model_update_loss =  self.alpha * weighted_task_loss * len(self.targets) - decision_loss

        # 移除返回字典键中的 "val_" 前缀，Keras 会自动添加
        return {
            "total_loss": model_update_loss,
            "weighted_task_loss": weighted_task_loss, 
            "decision_loss": decision_loss,
            "paid_loss": paid_loss, 
            "cost_loss": cost_loss,
        }