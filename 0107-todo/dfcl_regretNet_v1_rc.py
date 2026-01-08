import tensorflow as tf
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
import os
import sys
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

class EcomDFCL_regretNet_rc(tf.keras.Model):
    """
    使用 TensorFlow 2.x Keras API 实现的电商模型。
    该模型采用增广拉格朗日方法进行约束优化。
    只参考形式，但还没有完全还原。
    """
    def __init__(self, rho=0.1, dense_stats=None, fcd_mode='log1p', lambda_update_frequency=20, loss_function='2pll', max_multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        self.paid_pos_weight = 99.71/(100-99.71)
        self.cost_pos_weight=95.30/(100-95.30)
        
        # --- 增广拉格朗日方法超参数 ---
        self.rho = rho  # 二次惩罚项的系数 ρ
        self.lambda_update_frequency = lambda_update_frequency # 拉格朗日乘子 λ 的更新频率 Q
        self.max_multiplier = max_multiplier
        # 拉格朗日乘子，对应 paid_loss 和 cost_loss 两个约束
        self.lagrange_multipliers = tf.Variable([0.0, 0.0], trainable=False, name='lagrange_multipliers')

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
                local_loss += tf.reduce_sum(masked_loss)

            if target_name == 'paid':
                paid_loss += local_loss
            else:
                cost_loss += local_loss

        return paid_loss, cost_loss

    def decision_policy_learning_loss(self, predictions, labels): 
        pred_dict = {key: tf.exp(tf.minimum(logit, 10.0)) for key, logit in predictions.items()}
        decision_loss_sum = tf.constant(0.0, dtype=tf.float32)
        treatment_idx = tf.cast(labels['treatment'], tf.int32)
        treatment_masks = {
            t: tf.cast(tf.equal(treatment_idx, t), tf.float32) for t in self.treatment_order
        }
        N = self.total_samples
        counts_per_treatment = {
            t: float(self.treatment_sample_counts[t]) + 1e-8 for t in self.treatment_order
        }
        weight_tensor = tf.zeros_like(treatment_idx, dtype=tf.float32)
        for t in self.treatment_order:
            weight_for_t = N / counts_per_treatment[t]
            weight_tensor += treatment_masks[t] * weight_for_t
        weight_tensor = tf.reshape(weight_tensor, [-1, 1])
        for ratio in self.ratios:
            values = [
                pred_dict[f"paid_treatment_{t}"] - ratio * pred_dict[f"cost_treatment_{t}"]
                for t in self.treatment_order
            ]
            cancat_tensor = tf.nn.softmax(tf.stack(values, axis=1), axis=1)
            mask_tensor = tf.stack([treatment_masks[t] for t in self.treatment_order], axis=1)
            ratio_target = tf.reshape(labels['paid'] - ratio * labels['cost'], [-1, 1])
            decision_loss = tf.reduce_sum(cancat_tensor * mask_tensor * ratio_target * weight_tensor)

            decision_loss_sum += decision_loss
        return decision_loss_sum / len(self.ratios)
    
    # ... (其他 decision loss 函数保持不变) ...
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
                dot_product += tf.reduce_sum(g1 * g2)
                norm1_sq += tf.reduce_sum(g1 * g1)
                norm2_sq += tf.reduce_sum(g2 * g2)
        
        # 避免除以零
        denominator = tf.sqrt(norm1_sq * norm2_sq)
        return tf.math.divide_no_nan(dot_product, denominator)

    def train_step(self, data):
        """
        使用增广拉格朗日方法实现约束优化。
        - 主目标：最大化 decision_loss (即最小化 -decision_loss)
        - 约束：paid_loss 和 cost_loss 趋近于0
        """
        features, labels = data
        step = self.optimizer.iterations

        with tf.GradientTape() as tape:
            # 1. 前向传播，计算所有损失
            predictions = self(features, training=True)
            paid_loss, cost_loss = self.compute_local_losses(predictions, labels)
            decision_loss = self.decision_policy_learning_loss(predictions, labels)

            # 检查数值稳定性
            paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in paid_loss")
            cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in cost_loss")
            decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in decision_loss")

            # 2. 构建增广拉格朗日函数 L(w, λ) 用于更新模型参数 w
            # L(w, λ) = f(w) + λ^T * g(w) + (ρ/2) * ||g(w)||^2
            # f(w) 是主目标: -decision_loss (因为我们要最大化 decision_loss)
            # g(w) 是约束函数: [paid_loss, cost_loss] (我们希望它们为0)
            # λ 是拉格朗日乘子: self.lagrange_multipliers
            
            constraint_violations = tf.stack([paid_loss, cost_loss]) ############

            # 拉格朗日项: λ^T * g(w)
            # 在更新w时，λ被视为常数，因此停止梯度回传
            lambda_term = tf.reduce_sum(tf.stop_gradient(self.lagrange_multipliers) * constraint_violations)
            
            # 二次- NO 惩罚项: (ρ/2) * g(w)
            penalty_term = (self.rho / 2.0) * tf.reduce_sum(tf.square(constraint_violations))

            # 最终用于更新模型参数 w 的总损失
            model_update_loss = -decision_loss + lambda_term + penalty_term

        # 3. 计算梯度并更新模型参数 w (Primal Update)
        model_variables = self.trainable_variables
        model_gradients = tape.gradient(model_update_loss, model_variables)
        self.optimizer.apply_gradients(zip(model_gradients, model_variables))

        # 4. 每隔 Q 步，更新拉格朗日乘子 λ (Dual Update)
        def update_lambdas():
            # 更新规则: λ_new = λ_old + ρ * g(w)
            # 使用 tf.stop_gradient 确保此更新操作不会影响 w 的梯度计算
            # self.lagrange_multipliers.assign_add(self.rho * tf.stop_gradient(constraint_violations))
            new_lambdas = self.lagrange_multipliers + self.rho * tf.stop_gradient(constraint_violations)
            # 由于 multiplier 在训练过程中可能持续增大，这里对其施加上界（默认 10），并保证非负
            clipped_lambdas = tf.clip_by_value(new_lambdas, clip_value_min=0.0, clip_value_max=self.max_multiplier)
            self.lagrange_multipliers.assign(clipped_lambdas)
            return tf.constant(True)

        def no_lambda_update():
            return tf.constant(False)
        
        # 使用 tf.cond 实现条件更新
        tf.cond(
            tf.equal(step % self.lambda_update_frequency, 0),
            update_lambdas,
            no_lambda_update
        )

        # 5. 记录监控指标到 TensorBoard
        self._add_summaries("losses/1_paid_loss", paid_loss, step=step)
        self._add_summaries("losses/2_cost_loss", cost_loss, step=step)
        self._add_summaries("losses/3_decision_loss", decision_loss, step=step)
        self._add_summaries("losses/4_model_update_loss", model_update_loss, step=step)
        tf.summary.scalar("lagrangian/lambda_paid", self.lagrange_multipliers[0], step=step)
        tf.summary.scalar("lagrangian/lambda_cost", self.lagrange_multipliers[1], step=step)
        tf.summary.scalar("lagrangian/penalty_term", penalty_term, step=step)

        # 返回用于 Keras 进度条显示的指标
        return {
            "total_loss": model_update_loss,
            "decision_loss": decision_loss,
            "paid_loss": paid_loss,
            "cost_loss": cost_loss,
        }

    def test_step(self, data):
        """验证步骤，不更新权重，只计算损失"""
        features, labels = data
        predictions = self(features, training=False)

        paid_loss, cost_loss = self.compute_local_losses(predictions, labels)
        decision_loss = self.decision_policy_learning_loss(predictions, labels)
        
        # 在验证集上，我们通常只关心原始损失，不计算增广拉格朗日损失
        model_update_loss = -decision_loss # 可以只看主目标

        return {
            "total_loss": model_update_loss,
            "decision_loss": decision_loss,
            "paid_loss": paid_loss, 
            "cost_loss": cost_loss,
        }