# preprocess_boxcox.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import json
import tensorflow as tf

def find_optimal_transformations(tfrecord_files, dense_feature_names, 
                                  output_path='dense_transform_params.json'):
    """
    为每个 dense feature 寻找最优的 Box-Cox/Yeo-Johnson 参数
    """
    # 1. 加载数据样本（不需要全部数据，10万样本足够）
    feature_schema = {name: tf.io.FixedLenFeature([], tf.float32) for name in dense_feature_names}
    
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_schema))
    dataset = dataset.take(100000).batch(100000)
    
    sample_data = next(iter(dataset))
    df = pd.DataFrame({name: sample_data[name].numpy() for name in dense_feature_names})
    
    # 2. 为每个特征找最优变换
    transform_params = {}
    
    for col in dense_feature_names: 
        values = df[col].values.reshape(-1, 1)
        
        # 检查是否有负值（决定用 Box-Cox 还是 Yeo-Johnson）
        has_negative = (values < 0).any()
        
        if has_negative: 
            # Yeo-Johnson 支持负值
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            # Box-Cox 只支持正值，性能更好
            values = np.maximum(values, 1e-6)  # 避免0
            pt = PowerTransformer(method='box-cox', standardize=True)
        
        pt.fit(values)
        
        # 测试正态性（Shapiro-Wilk test）
        transformed = pt.transform(values).flatten()
        _, p_value = stats.shapiro(transformed[: 5000])  # 只测试5000个样本
        
        transform_params[col] = {
            'method': 'yeo-johnson' if has_negative else 'box-cox',
            'lambda': float(pt.lambdas_[0]),
            'mean': float(np.mean(transformed)),
            'std': float(np.std(transformed)),
            'shapiro_p_value': float(p_value),
            'is_normal': p_value > 0.05  # p > 0.05 表示接受正态分布假设
        }
        
        print(f"{col}: method={transform_params[col]['method']}, "
              f"lambda={transform_params[col]['lambda']:. 4f}, "
              f"normal={transform_params[col]['is_normal']}")
    
    with open(output_path, 'w') as f:
        json.dump(transform_params, f, indent=2)
    
    print(f"\n✅ 变换参数已保存到 {output_path}")
    return transform_params

# preprocess_boxcox.py 追加：从 tf.data.Dataset 抽样计算 Box-Cox/YJ 参数

def find_optimal_transformations_from_dataset(ds,
                                             dense_feature_names,
                                             output_path='dense_transform_params.json',
                                             max_samples=100000):
    """
    ds: tf.data.Dataset
        每个元素可以是：
          1) features(dict)
          2) (features(dict), labels)
    dense_feature_names: list[str]
    """
    buf = {k: [] for k in dense_feature_names}
    seen = 0

    for elem in ds:
        features = elem[0] if isinstance(elem, tuple) else elem

        # 取 batch size（假设 dense 特征都是 [B] 或 [B,1]）
        any_key = dense_feature_names[0]
        b = int(features[any_key].shape[0])

        for k in dense_feature_names:
            x = tf.cast(features[k], tf.float32)
            x = x.numpy()
            x = x.reshape(-1)  # 展平
            buf[k].append(x)

        seen += b
        if seen >= max_samples:
            break

    # 拼接并截断到 max_samples
    df = pd.DataFrame({
        k: np.concatenate(buf[k], axis=0)[:max_samples]
        for k in dense_feature_names
    })

    transform_params = {}

    for col in dense_feature_names:
        values = df[col].values.reshape(-1, 1)
        has_negative = (values < 0).any()

        if has_negative:
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            pt.fit(values)
            transform_params[col] = {
                "method": "yeo-johnson",
                "lambda": float(pt.lambdas_[0]),
                "mean": float(pt._scaler.mean_[0]),
                "std": float(pt._scaler.scale_[0])
            }
        else:
            # Box-Cox 需要严格正数：<=0 的做一个很小的平移
            min_val = values.min()
            shift = 0.0
            if min_val <= 0:
                shift = float(-min_val + 1e-6)
                values = values + shift

            pt = PowerTransformer(method='box-cox', standardize=True)
            pt.fit(values)
            transform_params[col] = {
                "method": "box-cox",
                "lambda": float(pt.lambdas_[0]),
                "mean": float(pt._scaler.mean_[0]),
                "std": float(pt._scaler.scale_[0]),
                "shift": shift
            }

    # 写出 json
    with tf.io.gfile.GFile(output_path, "w") as f:
        json.dump(transform_params, f, indent=2)

    return transform_params
