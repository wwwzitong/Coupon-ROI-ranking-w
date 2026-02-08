# python create_sample_features.py \
#   --train_data ../data/criteo_train.csv \
#   --val_data ../data/criteo_val.csv \
#   --batch_size 4096 \
#   --split val \
#   --out sample_features_criteo.npz \
#   --labels_out sample_labels_criteo.npz

python create_sample_features.py \
  --train_data ../data/ECLIFT_train.csv \
  --val_data ../data/ECLIFT_val.csv \
  --batch_size 4096 \
  --split val \
  --out sample_features_ECLIFT_4096.npz \
  --labels_out sample_labels_ECLIFT_4096.npz