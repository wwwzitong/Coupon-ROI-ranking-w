# python train_SL.py --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100


# python train.py --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw

# python train.py --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw

# python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p


# python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_batchmean_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw



# run2:

python train_SL.py --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_run2


python train.py --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_run2

python train.py --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_run2

python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p_run2


python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_batchmean_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_run2



python Evaluation.py








# test:
# python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-4 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-4_alpha=100_clip=100_log1p
# python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 512 --alldata False --lr 2e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs512_step500_lr2e-3_alpha=100_clip=100_log1p
