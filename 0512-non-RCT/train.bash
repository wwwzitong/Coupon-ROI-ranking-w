
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 2048 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_batchmean_bs2048_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw

python Evaluation.py
