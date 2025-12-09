# -*- encoding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import subprocess
import re
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
                self.output_types[name] = dtype
                self.output_shapes[name] = tf.TensorShape([])
    
    def _rename_columns(self, features):
        """
        重命名特定列
        conversion -> paid
        visit -> cost
        """
        # 注意：在Graph模式下，尽量避免原地修改(inplace)，虽然字典操作通常兼容
        # 但为了安全起见，显式地进行键值转移
        
        # 1. 检查是否存在 'conversion'，将其赋值给 'paid' 并删除原键
        if 'conversion' in features:
            features['paid'] = features.pop('conversion')
            
        # 2. 检查是否存在 'visit'，将其赋值给 'cost' 并删除原键
        if 'visit' in features:
            features['cost'] = features.pop('visit')
            
        return features

    def _parse_csv_line(self, line):
        """解析CSV单行数据"""
        # 将CSV行拆分为字段
        fields = tf.io.decode_csv(line, record_defaults=self.column_defaults)
        
        # 将字段打包为字典
        features = dict(zip(self.column_names, fields))
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
        
        # 执行列重命名逻辑
        dataset = dataset.map(self._rename_columns, num_parallel_calls=4)

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