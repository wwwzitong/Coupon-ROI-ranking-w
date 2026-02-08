
# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --rho 0.1 --model_path ./model/rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_rho=0.1
python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --rho 0.15 --model_path ./model/rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_rho=0.15


# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --rho 0.1 --model_path ./model/rplusc_wce_loss2_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_rho=0.1_ratiosum

# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --rho 0.01 --model_path ./model/rplusc_nopenalty_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_rho=0.01


# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.5 --bs 4096 --max_multiplier 0.1 --lr 1e-3 --clipnorm 5e3 --rho 0.01 --model_path ./model/rplusc_nopred_bs4096_lr1e-3_clip=5e3_max=0.1_tau=0.5_rho=0.01


python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1 --lr 1e-4 --clipnorm 5e3 --rho 0 --model_path ./model/rplusc_wce_bs256_lr1e-4_clip=5e3_max=1_tau=1_rho=0
python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 0 --lr 1e-4 --clipnorm 5e3 --rho 0 --model_path ./model/rplusc_wce_bs256_lr1e-4_clip=5e3_max=0_tau=1_rho=0


# python Evaluation.py