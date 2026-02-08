# python train_SL.py --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100


# python train.py --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw

# python train.py --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw

# python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p


python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 4096 --max_multiplier 0.08 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.08_tau=1.0_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 4096 --max_multiplier 0.12 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.12_tau=1.0_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 4096 --max_multiplier 0.1 --lr 5e-4 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr5e-4_clip=5e3_max=0.1_tau=1.0_log1p

python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.1 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.1_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.3 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.3_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=1.0_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.2 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=1.2_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=1.5_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.0 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=2.0_log1p
python train_our.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode log1p --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=2.5_log1p

python Evaluation_old.py

# run2:

# python train_SL.py --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_run2


# python train.py --seed 42 --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_run9

# python train.py --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_run2

# python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p_run2


# python train_our.py --seed 42 --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_run6



# seed 41:
# python train_SL.py --seed 41 --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_seed41
# python train_SL.py --seed 41 --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_seed41_run2


# python train.py --seed 41 --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_seed41

# python train.py --seed 41 --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_seed41

# python train.py --seed 41 --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p_seed41


# python train_our.py --seed 41 --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_seed41


# seed 43:
# python train_SL.py --seed 43 --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_seed43

# python train.py --seed 43 --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_seed43
# python train.py --seed 43 --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_seed43
# python train.py --seed 43 --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p_seed43

# python train_our.py --seed 43 --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_seed43


# seed 44:
# python train_SL.py --seed 44 --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_seed44

# python train.py --seed 44 --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_seed44
# python train.py --seed 44 --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_seed44
# python train.py --seed 44 --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p_seed44

# python train_our.py --seed 44 --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_seed44


# seed 40:
# python train_SL.py --seed 40 --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_seed40

# python train.py --seed 40 --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_seed40
# python train.py --seed 40 --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_seed40
# python train.py --seed 40 --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p_seed40

# python train_our.py --seed 40 --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_seed40


# seed 39:
# python train_SL.py --seed 39 --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100_seed39

# python train.py --seed 39 --model_class_name EcomDFCL_v3 --loss_function 2pll --bs 1024 --alldata False --lr 1e-3 --alpha 0.1 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_bs1024_step500_lr1e-3_alpha=0.1_clip=5e3_raw_seed39
# python train.py --seed 39 --model_class_name EcomDFCL_v3 --loss_function 3erl --bs 1024 --alldata False --lr 1e-3 --tau 3 --alpha 100 --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_3erl_bs1024_step500_lr1e-3_alpha=100_clip=5e3_tau=3_raw_seed39
# python train.py --seed 39 --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-3_alpha=100_clip=100_log1p_seed39

# python train_our.py --seed 39 --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --fcd_mode raw --model_path ./model/EcomDFCL_regretNet_rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_raw_seed39



# python Evaluation_old.py








# test:
# python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 256 --alldata False --lr 1e-4 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs256_step500_lr1e-4_alpha=100_clip=100_log1p
# python train.py --model_class_name EcomDFCL_v3 --loss_function 4ifdl --bs 512 --alldata False --lr 2e-3 --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_bs512_step500_lr2e-3_alpha=100_clip=100_log1p
