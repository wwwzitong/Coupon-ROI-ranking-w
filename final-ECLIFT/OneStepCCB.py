import tensorflow as tf
import os
import numpy as np
import sys

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_utils_ECLIFT import *

SPARSE_FEATURE_NAME = ['feat1_seq1', 'feat1_seq2', 'feat1_seq3', 'feat1_seq4',
                       'feat1_seq5', 'feat1_seq6', 'feat1_seq7', 'feat1_seq8', 'feat2_seq1']
DENSE_FEATURE_NAME = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
                      'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15']
SPARSE_FEATURE_NAME_SLOT_ID = {name: idx for idx, name in enumerate(SPARSE_FEATURE_NAME)}

statistical_config = {
    'N': 10000000,
    'N1': 7499644,
    'N0': 2500356
}


class EcomOneStepCCB(tf.keras.Model):
    """
    1-step constrained contextual bandit baseline
    - 输入仍然是单步样本 (features, labels)
    - 输出是 action logits / probs，而不是 potential outcome prediction
    - 训练使用 IPS policy objective
    """
    def __init__(
        self,
        entropy_coef=1e-3,
        ips_clip=10.0,
        batch_sum_mean='mean',
        dense_stats=None,
        fcd_mode='log1p',
        tau=1.0,
        use_ratio_grid=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.entropy_coef = entropy_coef
        self.ips_clip = ips_clip
        self.tau = tau

        # 是否像你原始 DFCL 一样，对多个 ratio 做平均
        self.use_ratio_grid = use_ratio_grid
        self.ratio_grid = [i / 100.0 for i in range(5, 105,5)]

        self.sparse_feature_names = SPARSE_FEATURE_NAME
        self.dense_feature_names = DENSE_FEATURE_NAME
        self.sparse_feature_slot_map = SPARSE_FEATURE_NAME_SLOT_ID
        self.num_estimated_vec_features = 10000

        self.fcd_mode = fcd_mode
        self._dense_global_mean = None
        self._dense_global_std = None
        if dense_stats is not None:
            stats_obj = dense_stats.get(self.fcd_mode, dense_stats)
            mean_list = stats_obj.get("mean")
            std_list = stats_obj.get("std")
            if mean_list is None or std_list is None:
                raise ValueError(f"dense_stats 缺少 mean/std: keys={list(stats_obj.keys())}")

            self._dense_global_mean = tf.convert_to_tensor(mean_list, dtype=tf.float32)
            self._dense_global_std = tf.convert_to_tensor(std_list, dtype=tf.float32)

        self.sparse_feature_dim = 8
        self.dense_feature_dim = 1
        self.sparse_pooling_type = "mean"

        # 行为策略概率（如果日志是随机分流，可直接用全局占比）
        self.behavior_prob_t1 = statistical_config['N1'] / statistical_config['N']
        self.behavior_prob_t0 = statistical_config['N0'] / statistical_config['N']

        self._build_feature_layers()

        # 共享表征层：沿用你现有风格
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
        ], name='user_tower')

        user_tower_input_dim = (
            len(self.sparse_feature_names) * self.sparse_feature_dim +
            len(self.dense_feature_names) * self.dense_feature_dim
        )
        self.user_tower.build(input_shape=(None, user_tower_input_dim))

        # policy head：直接输出两个 action 的 logits
        # action 0 = control, action 1 = treatment
        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(2, kernel_initializer='glorot_normal')
        ], name='policy_head')

    def _build_feature_layers(self):
        self.hashing_layers = {}
        for feature_name in self.sparse_feature_names:
            self.hashing_layers[feature_name] = tf.keras.layers.Hashing(
                num_bins=self.num_estimated_vec_features,
                name=f"hashing_{feature_name}"
            )

        self.embedding_layers = {}
        unique_slot_ids = set(self.sparse_feature_slot_map.values())
        for slot_id in unique_slot_ids:
            self.embedding_layers[str(slot_id)] = tf.keras.layers.Embedding(
                input_dim=self.num_estimated_vec_features,
                output_dim=self.sparse_feature_dim,
                name=f"embedding_slot_{slot_id}"
            )

        self.attn_score_layers = {}
        if self.sparse_pooling_type == "attention":
            for feature_name in self.sparse_feature_names:
                self.attn_score_layers[feature_name] = tf.keras.layers.Dense(
                    1, use_bias=False, name=f"attn_score_{feature_name}"
                )

    def _pool_sparse_embeddings(self, emb, mask, feature_name=None):
        mask_f = tf.cast(mask, tf.float32)
        mask_f_e = tf.expand_dims(mask_f, axis=-1)

        if self.sparse_pooling_type == "mean":
            emb_masked = emb * mask_f_e
            denom = tf.reduce_sum(mask_f_e, axis=1) + 1e-8
            return tf.reduce_sum(emb_masked, axis=1) / denom

        if self.sparse_pooling_type == "max":
            very_neg = tf.constant(-1e9, dtype=emb.dtype)
            emb_masked = tf.where(mask_f_e > 0, emb, very_neg)
            return tf.reduce_max(emb_masked, axis=1)

        score_layer = self.attn_score_layers[feature_name]
        scores = score_layer(emb)
        very_neg = tf.constant(-1e9, dtype=scores.dtype)
        scores = tf.where(mask_f_e > 0, scores, very_neg)
        attn = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(emb * attn, axis=1)

    def _build_concat_input(self, inputs, training=True):
        sparse_vectors = []

        for feature_name in self.sparse_feature_names:
            raw = inputs[feature_name]

            if raw.dtype == tf.string:
                mask = tf.logical_and(tf.not_equal(raw, ""), tf.not_equal(raw, "0"))
                ids = self.hashing_layers[feature_name](raw)
            else:
                raw_int = tf.cast(raw, tf.int32)
                mask = tf.not_equal(raw_int, 0)
                ids = tf.math.floormod(raw_int, self.num_estimated_vec_features)

            ids = tf.cast(ids, tf.int32)

            if ids.shape.rank == 1:
                ids = tf.expand_dims(ids, axis=1)
                mask = tf.expand_dims(mask, axis=1)
            else:
                ids = tf.cond(tf.equal(tf.rank(ids), 1),
                              lambda: tf.expand_dims(ids, 1),
                              lambda: ids)
                mask = tf.cond(tf.equal(tf.rank(mask), 1),
                               lambda: tf.expand_dims(mask, 1),
                               lambda: mask)

            slot_id = str(self.sparse_feature_slot_map[feature_name])
            emb = self.embedding_layers[slot_id](ids)
            pooled = self._pool_sparse_embeddings(emb, mask, feature_name=feature_name)
            sparse_vectors.append(pooled)

        B = tf.shape(inputs[self.dense_feature_names[0]])[0]
        dense_vectors = []

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

            dense_vectors.append(tf.reshape(fcd, [B, 1]))

        concat_input = tf.concat(sparse_vectors + dense_vectors, axis=1)
        return concat_input

    def call(self, inputs, training=True):
        concat_input = self._build_concat_input(inputs, training=training)

        x = concat_input
        user_tower_activations = {}
        for i, layer in enumerate(self.user_tower.layers):
            x = layer(x, training=training)
            user_tower_activations[f"layer_{i}_{layer.name}"] = x

        shared_output = x
        action_logits = self.policy_head(shared_output, training=training)   # [B, 2]
        action_prob = tf.nn.softmax(action_logits / self.tau, axis=-1)       # [B, 2]

        self._last_shared_output = shared_output
        self._last_user_tower_activations = user_tower_activations

        return {
            "action_logits": action_logits,
            "action_prob": action_prob,
            "prob_action_0": action_prob[:, 0],
            "prob_action_1": action_prob[:, 1],
        }

    def _compute_utility(self, labels, ratio):
        paid = tf.cast(labels['paid'], tf.float32)
        cost = tf.cast(labels['cost'], tf.float32)
        utility = paid - ratio * cost
        return utility

    def _compute_ips_policy_value(self, outputs, labels, ratio):
        """
        单步离线 bandit 目标：
            E[ pi(a|x) / mu(a|x) * u ]
        a 是日志里的真实 action=treatment
        u 是该 action 下观测到的 utility = paid - ratio * cost
        """
        action_prob = outputs["action_prob"]   # [B,2]
        treatment = tf.cast(labels['treatment'], tf.int32)

        batch_idx = tf.range(tf.shape(treatment)[0], dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, treatment], axis=1)
        pi_a = tf.gather_nd(action_prob, gather_idx)  # 当前策略给日志动作的概率

        mu_a = tf.where(
            tf.equal(treatment, 1),
            tf.ones_like(pi_a, dtype=tf.float32) * self.behavior_prob_t1,
            tf.ones_like(pi_a, dtype=tf.float32) * self.behavior_prob_t0
        )

        ips_weight = tf.math.divide_no_nan(pi_a, mu_a)
        ips_weight = tf.clip_by_value(ips_weight, 0.0, self.ips_clip)

        utility = self._compute_utility(labels, ratio)
        # policy_value = tf.reduce_mean(ips_weight * utility)
        policy_value = tf.reduce_sum(ips_weight * utility) / (tf.reduce_sum(ips_weight) + 1e-8)
        return policy_value, ips_weight, utility

    def _compute_entropy(self, outputs):
        prob = tf.clip_by_value(outputs["action_prob"], 1e-8, 1.0)
        entropy = -tf.reduce_sum(prob * tf.math.log(prob), axis=1)
        return tf.reduce_mean(entropy)

    def train_step(self, data):
        features, labels = data

        with tf.GradientTape() as tape:
            outputs = self(features, training=True)

            policy_value_sum = tf.constant(0.0, dtype=tf.float32)
            last_ips_weight = None
            last_utility = None

            for ratio in self.ratio_grid:
                policy_value, ips_weight, utility = self._compute_ips_policy_value(outputs, labels, ratio)
                policy_value_sum += policy_value
                last_ips_weight = ips_weight
                last_utility = utility

            avg_policy_value = policy_value_sum / float(len(self.ratio_grid))
            entropy = self._compute_entropy(outputs)

            # maximize policy value + entropy regularization
            total_loss = -avg_policy_value - self.entropy_coef * entropy

        variables = self.trainable_variables
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        step = self.optimizer.iterations
        tf.summary.scalar("losses/total_loss", total_loss, step=step)
        tf.summary.scalar("policy/avg_policy_value", avg_policy_value, step=step)
        tf.summary.scalar("policy/entropy", entropy, step=step)
        tf.summary.scalar("policy/avg_prob_action_1", tf.reduce_mean(outputs["prob_action_1"]), step=step)
        tf.summary.scalar("policy/avg_ips_weight", tf.reduce_mean(last_ips_weight), step=step)
        tf.summary.scalar("policy/avg_utility", tf.reduce_mean(last_utility), step=step)

        valid_grads = [g for g in grads if g is not None]
        if valid_grads:
            global_norm = tf.linalg.global_norm(valid_grads)
            tf.summary.scalar("gradients/global_norm", global_norm, step=step)

        return {
            "total_loss": total_loss,
            "avg_policy_value": avg_policy_value,
            "entropy": entropy,
            "avg_prob_action_1": tf.reduce_mean(outputs["prob_action_1"]),
            "avg_ips_weight": tf.reduce_mean(last_ips_weight),
            "avg_utility": tf.reduce_mean(last_utility),
        }

    def test_step(self, data):
        features, labels = data
        outputs = self(features, training=False)

        policy_value_sum = tf.constant(0.0, dtype=tf.float32)
        last_ips_weight = None
        last_utility = None

        for ratio in self.ratio_grid:
            policy_value, ips_weight, utility = self._compute_ips_policy_value(outputs, labels, ratio)
            policy_value_sum += policy_value
            last_ips_weight = ips_weight
            last_utility = utility

        avg_policy_value = policy_value_sum / float(len(self.ratio_grid))
        entropy = self._compute_entropy(outputs)
        total_loss = -avg_policy_value - self.entropy_coef * entropy

        return {
            "total_loss": total_loss,
            "avg_policy_value": avg_policy_value,
            "entropy": entropy,
            "avg_prob_action_1": tf.reduce_mean(outputs["prob_action_1"]),
            "avg_ips_weight": tf.reduce_mean(last_ips_weight),
            "avg_utility": tf.reduce_mean(last_utility),
        }