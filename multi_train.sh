#!/bin/bash

# --- 实验配置 ---
# 你想要训练的模型类名称
MODEL_CLASS="EcomDFCL_v3" 

# 用于保存模型的基础路径。脚本会自动在其后添加 _1, _2, ...
BASE_MODEL_PATH="./model_stability/ecom_DFCL_3entropy_2pos_std_tau1.2"

# 你希望运行实验的总次数
NUM_RUNS=5

# --- 循环运行 ---
echo "=================================================="
echo "开始为模型 [$MODEL_CLASS] 运行系列实验"
echo "总运行次数: $NUM_RUNS"
echo "=================================================="

for i in $(seq 1 $NUM_RUNS)
do
  # 为当前运行构建唯一的模型路径
  CURRENT_MODEL_PATH="${BASE_MODEL_PATH}_${i}"
  
  echo ""
  echo "--------------------------------------------------"
  echo ">>>>> 开始第 $i 次运行 (共 $NUM_RUNS 次)"
  echo ">>>>> 模型类: $MODEL_CLASS"
  echo ">>>>> 保存路径: $CURRENT_MODEL_PATH"
  echo "--------------------------------------------------"
  
  # 使用指定的参数执行 Python 脚本
  python train.py \
    --model_class_name "$MODEL_CLASS" \
    --model_path "$CURRENT_MODEL_PATH"
    
  # 检查脚本是否成功运行
  if [ $? -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "错误: 第 $i 次运行失败。"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # 如果希望在某次运行失败时中止整个脚本，可以取消下面这行的注释
    # exit 1
  fi
done

echo ""
echo "=================================================="
echo "所有 $NUM_RUNS 次实验均已完成。"
echo "=================================================="