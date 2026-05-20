
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 1024 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_batchmean_bs1024_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_1p0final_wopos

python train_SL.py --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_bs1024_lr3_clip=100_1p0final_wopos

python train.py --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_1p0final_wopos

python train.py --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_1p0final_wopos

python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 1024 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs1024_step500_lr1e-3_alpha=100_clip=100_log1p_1p0final_wopos

python Evaluation.py
