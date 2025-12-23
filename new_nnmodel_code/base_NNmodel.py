import tensorflow as tf
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

class basemodel(tf.keras.Model): # std+2pos
    """
    使用 TensorFlow 2.x Keras API 实现的电商模型。
    该模型集成了 GradNorm 和自定义决策损失。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        for target in self.targets:
            name = f"{target}_tower"
            self.task_towers[name] = tf.keras.Sequential([
                tf.keras.layers.Dense(dims, activation='relu', kernel_initializer='glorot_normal') for dims in tower_dims[:-1]
            ] + [
                tf.keras.layers.Dense(tower_dims[-1], kernel_initializer='glorot_normal')
            ], name=name)

        # GradNorm 的可训练损失权重
#         self.loss_weights = tf.Variable([1.0, 1.0], trainable=True, name='grad_norm_loss_weights')

    def call(self, inputs, training=True): # 定义数据从输入到输出的完整流动路径       
        # 1. 处理用户dense特征
        user_dense_vectors = []
        for feature_name in self.dense_feature_names:
            fcd = inputs[feature_name]
            fcd = tf.math.log1p(tf.maximum(fcd,0.0))
            fcd = (fcd - tf.reduce_mean(fcd)) / (tf.math.reduce_std(fcd) + 1e-8)
            user_dense_vectors.append(tf.reshape(fcd, [-1, self.dense_feature_dim]))

        dense_tower_input = tf.concat(user_dense_vectors, axis=1)
        dense_tower_output = self.dense_inputs_layer(dense_tower_input, training=training)
        
        # 为了监控，保存中间激活
        x = dense_tower_input
        user_tower_activations = {}
        for i, layer in enumerate(self.dense_inputs_layer.layers):
            x = layer(x, training=training)
            user_tower_activations[f"layer_{i}_{layer.name}"] = x
        self._last_user_tower_activations = user_tower_activations

        if training:
            # --- 训练路径 ---
            # 从输入数据中处理 treatment 特征
            treatment_dense_vectors = []
            for feature_name in self.treatment_dense_feature_names:
                # 假设 treatment 是一个浮点数特征 (e.g., 0.0 or 1.0)
                fcd = tf.cast(inputs[feature_name], tf.float32)
                treatment_dense_vectors.append(tf.reshape(fcd, [-1, self.dense_feature_dim]))
            
            treatment_vectors_shortcut = tf.concat(treatment_dense_vectors, axis=1)
            
            # 组合用户表征和 treatment shortcut
            shared_output = tf.concat([dense_tower_output, treatment_vectors_shortcut], axis=1)
            
            # 生成预测
            predictions = {}
            for name, tower in self.task_towers.items():
                pred_name = name.replace('_tower', '')
                logit = tower(shared_output, training=training)
                predictions[pred_name] = tf.reshape(logit, [-1])
            
            self._last_shared_output = shared_output
            return predictions

        else:
            # --- 推理路径: 反事实预测 ---
            # --- 推理路径: 使用固定的 treatment 进行反事实推理 ---
            flat_predictions = {}
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
                    logit = tower(shared_output, training=False)
                    # 构造 'gmv_treatment_1' 这样的键
                    flat_key = f"{pred_name}_treatment_{treatment_id}"
                    flat_predictions[flat_key] = tf.reshape(logit, [-1])

            return flat_predictions

    
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