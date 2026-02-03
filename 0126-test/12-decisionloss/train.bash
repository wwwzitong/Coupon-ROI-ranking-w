

# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 3 --lr 1e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_ratiomean_Nt_bs4096_lr1e-3_clip=5e3_max=1_tau=1.0

# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1.0 --lr 2e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_ratiomean_Nt_bs4096_lr2e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1.0 --lr 5e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_ratiomean_Nt_bs4096_lr5e-3_clip=5e3_max=1_tau=1.0

# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_ratiomean_Nt_bs4096_lr1e-4_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1.0 --lr 5e-4 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_ratiomean_Nt_bs4096_lr5e-4_clip=5e3_max=1_tau=1.0
python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1 --lr 1e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_mean_Nt_bs4096_lr1e-3_clip=5e3_max=1_tau=1.0_p2
python Evaluation.py

python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1 --lr 2e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_mean_Nt_bs4096_lr2e-3_clip=5e3_max=1_tau=1.0_p2
python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1 --lr 5e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_mean_Nt_bs4096_lr5e-3_clip=5e3_max=1_tau=1.0_p2
python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1 --lr 1e-4 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_mean_Nt_bs4096_lr1e-4_clip=5e3_max=1_tau=1.0_p2
python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1 --lr 5e-4 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_mean_Nt_bs4096_lr5e-4_clip=5e3_max=1_tau=1.0_p2
# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1.0 --lr 2e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_ratiomean_Nt_bs4096_lr2e-3_clip=5e3_max=1_tau=1.0
# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --scheduler raw --bs 4096 --max_multiplier 1.0 --lr 5e-3 --clipnorm 5e3 --model_path ./model/rplusc_wce_LSE_ratiomean_Nt_bs4096_lr5e-3_clip=5e3_max=1_tau=1.0


python Evaluation.py

