#!/bin/bash

# 定义要测试的分布列表
# 注意：beta 和 gamma 通常不直接用于残差(y-p)的损失，
# 这里主要对比 Bernoulli(Base), Normal(MSE), Laplace(MAE), Student-t(Robust)
DISTRIBUTIONS=("bernoulli")
SEEDS=(0 1 2)

# 基础参数
EPOCHS=50
BATCH_SIZE=256

echo "=========================================="
echo "开始 TODO 3 实验：残差分布假设验证"
echo "=========================================="

for dist in "${DISTRIBUTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Running Experiment: Dist=$dist, Seed=$seed"
        
        python train_todo3.py \
            --model_class_name EcomDFCL_v3 \
            --pred_loss_dist "$dist" \
            --seed "$seed" \
            --num_epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --do_residual_analysis \
            --residual_batches 100
        
        if [ $? -ne 0 ]; then
            echo "Error running experiment for $dist seed $seed"
            exit 1
        fi
    done
done

echo ""
echo "所有实验已完成。请查看 ./model/ 下各实验文件夹内的 residual_analysis_results 目录。"