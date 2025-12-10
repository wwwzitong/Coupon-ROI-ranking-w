#!/bin/bash

# 设置模型的基础保存目录
MODEL_BASE_DIR="./model_predreg_3entropy_lr4"
# 设置最大并行任务数，可以根据你的机器核心数来调整，例如 4 或 8
MAX_PARALLEL_JOBS=4

# 如果基础目录不存在，则创建它
mkdir -p $MODEL_BASE_DIR

echo "Starting hyperparameter tuning for alpha with up to $MAX_PARALLEL_JOBS parallel jobs..."

# 外层循环：遍历 alpha 值，从 0.0 到 1.0，步长为 0.1
for alpha_val in $(awk 'BEGIN{for(i=0.0; i<=1.0; i+=0.1) print i}')
do
  # 内层循环：每个 alpha 值训练 3 次
  for i in {1..3}
  do
    # 构造一个唯一的模型目录后缀和完整路径
    MODEL_SUFFIX="alpha${alpha_val}_run${i}"
    FULL_MODEL_PATH="${MODEL_BASE_DIR}/${MODEL_SUFFIX}"
    
    echo "----------------------------------------------------"
    echo "Queueing training: alpha=${alpha_val}, run=${i}"
    echo "Model will be saved in: ${FULL_MODEL_PATH}"
    echo "----------------------------------------------------"
    
    # 将训练任务放到后台执行，并在命令后添加 "&"
    python train.py \
      --model_class_name EcomDFCL_v3 \
      --alpha ${alpha_val} \
      --model_path ${FULL_MODEL_PATH} &
      
    # 当后台任务数量达到上限时，等待任一任务完成
    # `jobs -p` 会列出所有后台任务的PID
    # `wc -l` 会计算行数，即当前任务数
    while [[ $(jobs -p | wc -l) -ge $MAX_PARALLEL_JOBS ]]; do
      # 等待任意一个后台任务结束
      wait -n
    done
  done
done

# 等待所有剩余的后台任务全部完成
echo "Waiting for all remaining jobs to complete..."
wait

echo "All experiments finished."