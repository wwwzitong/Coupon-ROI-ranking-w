import tensorflow as tf
import json
#from fsfc_mine import * #自行生成fsfc文件（脚本放在data_flow中）
from data_utils import *
DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
statistical_config={
    'N':11183673,
    'N1':9506123,
    'N0':1677550
}
# 重要的 固定的treatment features，用于反事实推理
fixed_treatment_vec_blocks = {
    0: {'treatment': '0.0'},
    1: {'treatment': '1.0'},
}# e.g. fixed_treatment_vec_blocks[treatment]


class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # 这里的 units 决定了输出维度，实现降维 (e.g., 129 -> 64)
        self.linear = tf.keras.layers.Dense(self.units, activation=None, kernel_initializer='glorot_normal')
        self.gate = tf.keras.layers.Dense(self.units, activation='sigmoid', kernel_initializer='glorot_normal')
        super(GatedLinearUnit, self).build(input_shape)

    def call(self, inputs):
        return self.linear(inputs) * self.gate(inputs)


class GLU_base_DFCL(tf.keras.Model): # std+2pos
    """
    使用 TensorFlow 2.x Keras API 实现的电商模型。
    该模型集成了 GradNorm 和自定义决策损失。
    """
    def __init__(self, transform_params_path='dense_transform_params.json', **kwargs):
        # 如果传入了 alpha 就取出来，同时从 kwargs 里删掉它；没传就用默认值 1.0
        self.alpha = kwargs.pop('alpha', 0.1)
        self.loss_function = kwargs.pop('loss_function', "2pll")
        super().__init__(**kwargs)

        # 加载变换参数
        with open(transform_params_path, 'r') as f:
            self.transform_params = json.load(f)

        self.paid_pos_weight = 99.71/(100-99.71)
        self.cost_pos_weight= 95.30/(100-95.30)

        self.dense_feature_names = DENSE_FEATURE_NAME
        self.treatment_dense_feature_names = ['treatment'] # 'treatment' 是一个dense feature

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
        
        # 网络结构，dense_inputs_layer为用户特征的共享底层
        self.dense_inputs_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'), # 512->128
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),# 256->64
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal')# 128->32
        ], name='dense_inputs_layer')
        
        # 每个target一个独立的task tower
        self.task_towers = {} # 2x1，为prediction loss服务
        tower_dims = [64, 32, 1]
        # for target in self.targets:
        #     name = f"{target}_tower"
        #     self.task_towers[name] = tf.keras.Sequential([
        #         tf.keras.layers.Dense(dims, activation='relu', kernel_initializer='glorot_normal') for dims in tower_dims[:-1]
        #     ] + [
        #         tf.keras.layers.Dense(tower_dims[-1], kernel_initializer='glorot_normal')
        #     ], name=name)
        for target in self.targets:
            name = f"{target}_tower"
            self.task_towers[name] = tf.keras.Sequential([
                # 第一层：129 -> 64
                GatedLinearUnit(units=tower_dims[0]),
                tf.keras.layers.LayerNormalization(), # GLU 配合 LN 效果极佳
                tf.keras.layers.Dropout(0.1),
                
                # 第二层：64 -> 32
                GatedLinearUnit(units=tower_dims[1]),
                tf.keras.layers.LayerNormalization(),
                
                # 输出层：32 -> 1
                tf.keras.layers.Dense(tower_dims[-1], kernel_initializer='glorot_normal')
            ], name=name)


        # GradNorm 的可训练损失权重
#         self.loss_weights = tf.Variable([1.0, 1.0], trainable=True, name='grad_norm_loss_weights')

    def box_cox_transform(self, x, lmbda):
        """TensorFlow 实现的 Box-Cox 变换"""
        epsilon = 1e-6
        x = tf.maximum(x, epsilon)  # 确保正值
        
        if abs(lmbda) < 1e-6:
            return tf.math.log(x)
        else:
            return (tf.pow(x, lmbda) - 1.0) / lmbda
    
    def yeo_johnson_transform(self, x, lmbda):
        """TensorFlow 实现的 Yeo-Johnson 变换"""
        epsilon = 1e-6
        
        # Case 1: x >= 0, lambda != 0
        case1 = (tf.pow(x + 1, lmbda) - 1) / lmbda
        # Case 2: x >= 0, lambda == 0
        case2 = tf.math.log(x + 1)
        # Case 3: x < 0, lambda != 2
        case3 = -(tf.pow(-x + 1, 2 - lmbda) - 1) / (2 - lmbda)
        # Case 4: x < 0, lambda == 2
        case4 = -tf.math.log(-x + 1)
        
        result = tf.where(
            x >= 0,
            tf.where(tf.abs(lmbda) < epsilon, case2, case1),
            tf.where(tf.abs(lmbda - 2) < epsilon, case4, case3)
        )
        return result

    def call(self, inputs, training=True): # 定义数据从输入到输出的完整流动路径       
        # 1. 处理用户dense特征
        user_dense_vectors = []
        for feature_name in self.dense_feature_names:
            # fcd变换之前记录数据分布
            raw_val = inputs[feature_name]
            if training:
                if not hasattr(self, "_last_raw_inputs"):
                    self._last_raw_inputs = {}
                # 保存原始分布 (展平以便计算统计量)
                self._last_raw_inputs[feature_name] = tf.reshape(raw_val, [-1])

            fcd = inputs[feature_name]

            params = self.transform_params[feature_name]

            # 应用变换
            if params['method'] == 'box-cox':
                fcd = self.box_cox_transform(fcd, params['lambda'])
            else:  # yeo-johnson
                fcd = self.yeo_johnson_transform(fcd, params['lambda'])
            
            # 标准化
            fcd = (fcd - params['mean']) / (params['std'] + 1e-8)
            
            user_dense_vectors.append(tf.reshape(fcd, [-1, self.dense_feature_dim]))

            # fcd = tf.math.log1p(tf.maximum(fcd,0.0))
            # fcd = (fcd - tf.reduce_mean(fcd)) / (tf.math.reduce_std(fcd) + 1e-8)
            # user_dense_vectors.append(tf.reshape(fcd, [-1, self.dense_feature_dim]))

            # fcd变换之后记录数据分布
            if training:
                if not hasattr(self, "_last_fcd"):
                    self._last_fcd = {}
                self._last_fcd[feature_name] = tf.reshape(fcd, [-1])  # 展平成一维便于 histogram

        dense_tower_input = tf.concat(user_dense_vectors, axis=1)
        dense_tower_output = self.dense_inputs_layer(dense_tower_input, training=training)
        
        # 为了监控，保存中间激活
        x = dense_tower_input
        user_tower_activations = {}
        for i, layer in enumerate(self.dense_inputs_layer.layers):
            x = layer(x, training=training)
            user_tower_activations[f"layer_{i}_{layer.name}"] = x
        self._last_user_tower_activations = user_tower_activations

        # --- 对每个 treatment 进行反事实预测 (训练和推理通用) ---
        all_predictions = {}
        batch_size = tf.shape(next(iter(inputs.values())))[0]

        # 遍历固定的 treatment
        for treatment_id, treatment_features in fixed_treatment_vec_blocks.items():
            treatment_dense_vectors = []
            for feature_name in self.treatment_dense_feature_names:
                feature_value = float(treatment_features[feature_name])
                feature_tensor = tf.tile(tf.constant([[feature_value]], dtype=tf.float32), [batch_size, 1])
                treatment_dense_vectors.append(tf.reshape(feature_tensor, [-1, self.dense_feature_dim]))
            
            treatment_vectors_shortcut = tf.concat(treatment_dense_vectors, axis=1)
            shared_output = tf.concat([dense_tower_output, treatment_vectors_shortcut], axis=1)

             # 为当前 treatment 生成预测并展平到单个字典中
            for name, tower in self.task_towers.items():
                pred_name = name.replace('_tower', '')
                logit = tower(shared_output, training=training)
                # 构造 'paid_treatment_1' 这样的键
                flat_key = f"{pred_name}_treatment_{treatment_id}"
                all_predictions[flat_key] = tf.reshape(logit, [-1])
        
        self._last_shared_output = shared_output # Note: this will be the shared_output of the last treatment in the loop

        if training:
            # 训练时，需要分离出事实（factual）预测用于计算 prediction loss
            is_treatment_1 = tf.equal(tf.cast(inputs['treatment'], tf.int32), 1)
            
            factual_predictions = {}
            for target in self.targets:  # 'paid', 'cost'
                preds_if_treatment_1 = all_predictions[f'{target}_treatment_1']
                preds_if_treatment_0 = all_predictions[f'{target}_treatment_0']

                # 使用 tf.where 根据 treatment 指示器从两组预测中选择
                factual_pred_for_target = tf.where(
                    is_treatment_1,
                    preds_if_treatment_1,
                    preds_if_treatment_0
                )
                factual_predictions[target] = factual_pred_for_target
            
            # 返回所有预测（用于decision loss）和事实预测（用于prediction loss）
            return {'all': all_predictions, 'factual': factual_predictions}
        else:
            # 推理时，只需要所有反事实预测
            return all_predictions

    
    def compute_local_losses(self, predictions, labels): #和论文不一致啊
        paid_loss = tf.constant(0.0, dtype=tf.float32)
        cost_loss = tf.constant(0.0, dtype=tf.float32)

        for target_name in self.targets:
            if target_name == 'paid':
                pos_weight = self.paid_pos_weight
            else:
                pos_weight = self.cost_pos_weight
            
            logit = predictions[target_name]
            label = tf.cast(labels[target_name], tf.float32) 

            out = tf.minimum(logit, 10.0)
            term1 = -label * out
            term2 = (1 + label) * (tf.maximum(out, 0) + tf.math.log(1 + tf.exp(-tf.abs(out))))
            loss_per_sample = term1 + term2

            # 根据 pos_weight 对正样本的损失进行加权
            sample_weights = tf.where(label > 0, pos_weight, 1.0)
            weighted_loss_per_sample = loss_per_sample * sample_weights
            
            local_loss = tf.reduce_sum(weighted_loss_per_sample)

            if target_name == 'paid':
                paid_loss += local_loss
            else:
                cost_loss += local_loss

        return paid_loss, cost_loss

    def decision_policy_learning_loss(self, predictions, labels): #decision loss也和论文有出入
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
        tau = self.tau if hasattr(self, 'tau') else 1.5
        
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

        # ===== 新增：偏度 & 超额峰度 =====
        x = tf.cast(tf.reshape(tensor, [-1]), tf.float32)
        mean = tf.reduce_mean(x)
        std = tf.math.reduce_std(x) + 1e-8
        centered = x - mean

        m3 = tf.reduce_mean(tf.pow(centered, 3))
        m4 = tf.reduce_mean(tf.pow(centered, 4))

        skewness = m3 / tf.pow(std, 3)
        excess_kurtosis = m4 / tf.pow(std, 4) - 3.0

        tf.summary.scalar(f"{name}/skewness", skewness, step=step)
        tf.summary.scalar(f"{name}/excess_kurtosis", excess_kurtosis, step=step)


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
            # 1. 前向传播，获取所有反事实预测和事实预测
            predictions_dict = self(features, training=True)
            factual_predictions = predictions_dict['factual']
            all_predictions = predictions_dict['all']
            
            # 从模型属性中获取中间激活值
            shared_output = self._last_shared_output
            user_tower_activations = self._last_user_tower_activations
            
            # --- 进阶监控 3: 主动错误检测 ---
            for name, pred in factual_predictions.items():
                factual_predictions[name] = tf.debugging.check_numerics(pred, f"NaN/Inf in factual prediction: {name}")
            for name, pred in all_predictions.items():
                all_predictions[name] = tf.debugging.check_numerics(pred, f"NaN/Inf in counterfactual prediction: {name}")


            # 将模型的所有可训练变量分为两组：
            # 1. 模型参数（如 Tower、Embedding 等）
            # 2. GradNorm 的损失权重
            model_variables = [v for v in self.trainable_variables if 'loss_weights' not in v.name]

            paid_loss, cost_loss = self.compute_local_losses(factual_predictions, labels)
            # decision_loss = self.decision_improved_finite_difference_loss(all_predictions, labels) # TODO：loss输出有点奇怪
            if self.loss_function == "2pll":
                decision_loss = self.decision_policy_learning_loss(all_predictions, labels)
            elif self.loss_function == "3erl":
                decision_loss = self.decision_entropy_regularized_loss(all_predictions, labels)
            else:
                print("Error: loss function未定义")
            
            # --- 进阶监控 3 (续): 检查最终loss ---
            paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in paid_loss")
            cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in cost_loss")
            decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in decision_loss")
            
            # 2. 计算用于更新【模型参数】的损失
            # weighted_task_loss = 0.5 * paid_loss + 0.5 * cost_loss
            weighted_task_loss = paid_loss + cost_loss
            # 对应您代码中的 total_loss
            # model_update_loss = weighted_task_loss * len(self.targets) - decision_loss
            model_update_loss = self.alpha * weighted_task_loss - decision_loss
        

        # 4. 在 tape 上下文之外，分别计算多组梯度
        # 4.1. 计算用于更新模型的总梯度
        model_gradients = tape.gradient(model_update_loss, model_variables)
        
        # 4.2. 单独计算 task_loss 的梯度，用于分析
        # task_loss_gradients = tape.gradient(weighted_task_loss, model_variables)
        task_loss_gradients = tape.gradient(weighted_task_loss, model_variables)

        # 4.3. 单独计算 decision_loss 的梯度，用于分析
        decision_loss_gradients = tape.gradient(decision_loss, model_variables)
                
        # 5. 应用总梯度更新模型参数
        # 注意：这里使用 model_variables 而不是 self.trainable_variables，以确保梯度和变量列表匹配
        self.optimizer.apply_gradients(zip(model_gradients, model_variables))
        
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
        for name, pred_logit in factual_predictions.items():
            self._add_summaries(f"predictions/factual_{name}_logit", pred_logit, step=step)
        for name, pred_logit in all_predictions.items():
            self._add_summaries(f"predictions/counterfactual_{name}_logit", pred_logit, step=step)

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
        
        # 获取所有treatment下的反事实预测
        all_predictions = self(features, training=False)

        # 基于真实treatment数据，抽取出事实预测
        # 假设 'treatment' 在 features 中，并且其值为 0 或 1
        is_treatment_1 = tf.equal(tf.cast(features['treatment'], tf.int32), 1)
        
        factual_predictions = {}
        for target in self.targets:  # 'paid', 'cost'
            preds_if_treatment_1 = all_predictions[f'{target}_treatment_1']
            preds_if_treatment_0 = all_predictions[f'{target}_treatment_0']

            # 使用 tf.where 根据 treatment 指示器从两组预测中选择
            factual_pred_for_target = tf.where(
                is_treatment_1,
                preds_if_treatment_1,
                preds_if_treatment_0
            )
            factual_predictions[target] = factual_pred_for_target
        
        for name, pred in factual_predictions.items():
            factual_predictions[name] = tf.debugging.check_numerics(pred, f"NaN/Inf in validation prediction: {name}")

        paid_loss, cost_loss = self.compute_local_losses(factual_predictions, labels)
        if self.loss_function == "2pll":
            decision_loss = self.decision_policy_learning_loss(all_predictions, labels)
        elif self.loss_function == "3erl":
            decision_loss = self.decision_entropy_regularized_loss(all_predictions, labels)
        else:
            print("Error: loss function未定义")
        
        paid_loss = tf.debugging.check_numerics(paid_loss, "NaN/Inf in val_paid_loss")
        cost_loss = tf.debugging.check_numerics(cost_loss, "NaN/Inf in val_cost_loss")
        decision_loss = tf.debugging.check_numerics(decision_loss, "NaN/Inf in val_decision_loss")

        # weighted_task_loss = 0.5 * paid_loss + 0.5 * cost_loss
        # model_update_loss =  weighted_task_loss * len(self.targets) - decision_loss
        weighted_task_loss = paid_loss + cost_loss
        model_update_loss = self.alpha * weighted_task_loss - decision_loss

        # 移除返回字典键中的 "val_" 前缀，Keras 会自动添加
        return {
            "total_loss": model_update_loss,
            "weighted_task_loss": weighted_task_loss, 
            "paid_loss": paid_loss, 
            "cost_loss": cost_loss,
            "decision_loss": decision_loss,
        }

