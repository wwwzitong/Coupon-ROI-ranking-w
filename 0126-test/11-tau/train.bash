


python train.py --model_class_name EcomDFCL_regretNet_tau --max_multiplier 1.0 --scheduler raw --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_mean+ratios_bs4096_lr1e-4_clip=5e3_max=1
python train.py --model_class_name EcomDFCL_regretNet_tau --max_multiplier 1.0 --scheduler raw --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_mean+ratios_bs4096_lr5e-4_clip=5e3_max=1
python train.py --model_class_name EcomDFCL_regretNet_tau --max_multiplier 1.0 --scheduler raw --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_mean+ratios_bs4096_lr1e-3_clip=5e3_max=1
python train.py --model_class_name EcomDFCL_regretNet_tau --max_multiplier 1.0 --scheduler raw --lr 2e-3 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_mean+ratios_bs4096_lr2e-3_clip=5e3_max=1
python train.py --model_class_name EcomDFCL_regretNet_tau --max_multiplier 1.0 --scheduler raw --lr 5e-5 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_mean+ratios_bs4096_lr5e-5_clip=5e3_max=1




python Evaluation.py
