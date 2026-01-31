# -*- encoding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import subprocess
import re

algo_conf = {
# 清除以脱敏信息
}



# class TFRecordData:
#     def __init__(self):
#         self.data_conf = algo_conf['dataset']
#         self._generate_example_parser()

#     def _generate_example_parser(self):
#         dtype_map = {
#             "int": tf.int64,
#             "float": tf.float32,
#             "string": tf.string
#         }
#         default_map = {
#             "int": 0,
#             "float": 0.0,
#             "string": ""
#         }
#         example_parse_dict = {}
#         for column in self.data_conf["columns"]:
#             for name in column["names"]:
#                 default_value = column['default_value'] if 'default_value' in column else default_map[column['dtype']]
#                 example_parse_dict[name] = tf.io.FixedLenFeature(shape=[], dtype=dtype_map[column['dtype']],
#                                                                  default_value=default_value)
#         self.example_parse_dict = example_parse_dict

#     def prepare_dataset(self,
#                         sample_path,
#                         phase='train',
#                         threshold=None,
#                         batch_size=1024,
#                         shuffle=False,
#                         shuffle_buffer=2048):

#         def _parse_func(example_proto):
#             parsed_example = tf.io.parse_single_example(serialized=example_proto, features=self.example_parse_dict)
#             if threshold is not None and phase == 'predict':
#                 parsed_example['threshold'] = tf.ones_like(parsed_example['threshold']) * threshold
#             return parsed_example

#         # Generate Sample-Batch
#         # files will be shuffled randomly, default as True
#         dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(sample_path, shuffle=True))
#         dataset = dataset.map(_parse_func, num_parallel_calls=4)#tf.data.AUTOTUNE导致内存不足
#         dataset = dataset.shuffle(buffer_size=shuffle_buffer) if shuffle and phase == 'train' else dataset
#         dataset = dataset.batch(batch_size, drop_remainder=True if phase == 'train' else False)
#         dataset = dataset.prefetch(1)
#         return dataset

# -*- encoding=utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

algo_conf = {
    'dataset': {
        'columns': [
            {
                'names': ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11'],
                'dtype': 'float',
                'default_value': 0.0
            },
            {
                'names': ['treatment', 'conversion', 'visit', 'exposure'],
                'dtype': 'float',
                'default_value': 0.0
            }
        ]
    }
}

class CSVData:
    def __init__(self):
        self.data_conf = algo_conf['dataset']
        self.rename_map = {'conversion': 'paid', 'visit': 'cost'}
        self._generate_feature_spec()
        
    def _generate_feature_spec(self):
        """生成CSV列的数据类型和默认值配置"""
        dtype_map = {
            "int": tf.int64,
            "float": tf.float32,
            "string": tf.string
        }
        default_map = {
            "int": 0,
            "float": 0.0,
            "string": ""
        }
        
        self.column_defaults = []
        self.column_names = []
        self.output_types = {}
        self.output_shapes = {}
        
        for column in self.data_conf["columns"]:
            for name in column["names"]:
                default_value = column.get('default_value', default_map[column['dtype']])
                dtype = dtype_map[column['dtype']]
                
                self.column_names.append(name)
                self.column_defaults.append([default_value])
                output_name = self.rename_map.get(name, name)
                self.output_types[output_name] = dtype
                self.output_shapes[output_name] = tf.TensorShape([])
    
    def _parse_csv_line(self, line):
        """解析CSV单行数据"""
        # 将CSV行拆分为字段
        fields = tf.io.decode_csv(line, record_defaults=self.column_defaults)
        
        # 将字段打包为字典
        features = dict(zip(self.column_names, fields))

        # 根据rename_map重命名列
        for old_name, new_name in self.rename_map.items():
            if old_name in features:
                features[new_name] = features.pop(old_name)
        return features
    
    def prepare_dataset(self,
                        sample_path,
                        phase='train',
                        threshold=None,
                        batch_size=1024,
                        shuffle=False,
                        shuffle_buffer=2048):
        """
        准备数据集
        Args:
            sample_path: CSV文件路径或文件模式
            phase: 阶段 ('train', 'predict')
            threshold: 阈值（仅predict阶段使用）
            batch_size: 批次大小
            shuffle: 是否打乱数据
            shuffle_buffer: 打乱缓冲区大小
        Returns:
            tf.data.Dataset 对象
        """
        # 获取文件列表
        file_patterns = tf.io.gfile.glob(sample_path)
        if not file_patterns:
            raise ValueError(f"No files found matching pattern: {sample_path}")
        
        # 创建数据集
        dataset = tf.data.TextLineDataset(file_patterns)
        
        # 跳过CSV头部（如果有）
        dataset = dataset.skip(1)
        
        # 解析CSV数据
        dataset = dataset.map(self._parse_csv_line, num_parallel_calls=4)
        
        # 添加阈值（仅predict阶段）
        if threshold is not None and phase == 'predict':
            dataset = dataset.map(
                lambda x: {**x, 'threshold': tf.ones_like(x['treatment']) * threshold},
                num_parallel_calls=4
            )
        
        # 打乱数据
        if shuffle and phase == 'train':
            dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        
        # 批次处理
        dataset = dataset.batch(batch_size, drop_remainder=(phase == 'train'))
        
        # 预取
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
