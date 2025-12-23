#!/usr/bin/env bash
set -euo pipefail

# TODO 3: 从「分布假设 -> 训练(3次) -> 残差分布图」的一键流程
#
# 用法：
#   bash run_todo3_sweep.sh
#
# 你也可以只跑某几个分布：
#   DISTS=("bernoulli" "normal") bash run_todo3_sweep.sh

ALPHA="${ALPHA:-0.1}"
LOSS_FUNC="${LOSS_FUNC:-2pll}"
MODEL_CLASS="${MODEL_CLASS:-GLU_base_DFCL}"
TRAIN_DATA="${TRAIN_DATA:-data/criteo_train.csv}"
VAL_DATA="${VAL_DATA:-data/criteo_val.csv}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-0.0001}"
RESIDUAL_BATCHES="${RESIDUAL_BATCHES:-200}"

# 默认扫一组分布（可自行增删）
DISTS_DEFAULT=("bernoulli" "normal" "laplace" "t" "beta" "gamma")
# DISTS_DEFAULT=("bernoulli")
DISTS=("${DISTS[@]:-${DISTS_DEFAULT[@]}}")

SEEDS=(0 1 2)

for dist in "${DISTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    MODEL_PATH="./model/${MODEL_CLASS}_${LOSS_FUNC}_alpha=${ALPHA}_predloss=${dist}_seed=${seed}"
    echo "================================================================================"
    echo "[RUN] dist=${dist} seed=${seed}"
    echo "[RUN] model_path=${MODEL_PATH}"
    echo "================================================================================"

    python train_todo3.py \
      --model_class_name "${MODEL_CLASS}" \
      --model_path "${MODEL_PATH}" \
      --loss_function "${LOSS_FUNC}" \
      --alpha "${ALPHA}" \
      --pred_loss_dist "${dist}" \
      --seed "${seed}" \
      --train_data "${TRAIN_DATA}" \
      --val_data "${VAL_DATA}" \
      --batch_size "${BATCH_SIZE}" \
      --num_epochs "${EPOCHS}" \
      --learning_rate "${LR}" \
      --do_residual_analysis \
      --residual_batches "${RESIDUAL_BATCHES}"
  done
done

echo "✅ DONE. 请到 ./model/**/residual_analysis_predloss=.../ 查看残差图 & 报告"
