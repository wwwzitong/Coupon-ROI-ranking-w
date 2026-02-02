
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 8192 --max_multiplier 1.0 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs8192_lr5e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 8192 --max_multiplier 1.0 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs8192_lr1e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_v3 --tau 1.0 --bs 8192 --max_multiplier 1.0 --lr 2e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_mean+ratios_bs8192_lr2e-3_clip=5e3_max=1_tau=1.0

# python train.py --alpha 10 --loss_function 2pll --model_class_name EcomDFCL_v3 --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10
# python train.py --alpha 0.1 --loss_function 3erl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5

python train.py --alpha 100 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=5e3_alpha=100
python train.py --alpha 10 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=5e3_alpha=10
python train.py --alpha 2.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=5e3_alpha=2.5
python train.py --alpha 1.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=5e3_alpha=1.5
python train.py --alpha 1 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=5e3_alpha=1
python train.py --alpha 0.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=5e3_alpha=0.5

python train.py --alpha 100 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr5e-4_clip=5e3_alpha=100
python train.py --alpha 10 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr5e-4_clip=5e3_alpha=10
python train.py --alpha 2.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr5e-4_clip=5e3_alpha=2.5
python train.py --alpha 1.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr5e-4_clip=5e3_alpha=1.5
python train.py --alpha 1 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr5e-4_clip=5e3_alpha=1
python train.py --alpha 0.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr5e-4_clip=5e3_alpha=0.5

python train.py --alpha 100 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-4_clip=5e3_alpha=100
python train.py --alpha 10 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-4_clip=5e3_alpha=10
python train.py --alpha 2.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-4_clip=5e3_alpha=2.5
python train.py --alpha 1.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-4_clip=5e3_alpha=1.5
python train.py --alpha 1 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-4_clip=5e3_alpha=1
python train.py --alpha 0.5 --loss_function 4ifdl --model_class_name EcomDFCL_v3 --bs 512 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-4_clip=5e3_alpha=0.5



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

