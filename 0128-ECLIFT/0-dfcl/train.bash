
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 8192 --max_multiplier 1.0 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs8192_lr5e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 8192 --max_multiplier 1.0 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs8192_lr1e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 8192 --max_multiplier 1.0 --lr 2e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs8192_lr2e-3_clip=5e3_max=1_tau=1.0

python train.py --alpha 1.0 --loss_function 2pll --model_class_name EcomDFCL_v3 --bs 4096 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_2pll_mean+ratios_bs4096_lr1e-3_clip=5e3_alpha=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 4096 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs4096_lr1e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 4096 --max_multiplier 1.0 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs4096_lr5e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 4096 --max_multiplier 1.0 --lr 2e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs4096_lr2e-3_clip=5e3_max=1_tau=1.0

# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 2048 --max_multiplier 1.0 --lr 2e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs2048_lr2e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 2048 --max_multiplier 1.0 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs2048_lr1e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 2048 --max_multiplier 1.0 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs2048_lr5e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 2048 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs2048_lr1e-4_clip=5e3_max=1_tau=1.0

# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 1024 --max_multiplier 1.0 --lr 2e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs1024_lr2e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 1024 --max_multiplier 1.0 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs1024_lr1e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 1024 --max_multiplier 1.0 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs1024_lr5e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 1024 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs1024_lr1e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 1024 --max_multiplier 1.0 --lr 5e-5 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs1024_lr5e-5_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 1024 --max_multiplier 1.0 --lr 1e-5 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs1024_lr1e-5_clip=5e3_max=1_tau=1.0

python Evaluation.py

