
@REM v2_tau
@REM lr4
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_lr4_clip=5e3_log1p_max=100_rho2_sum
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_lr4_clip=100_log1p_max=100_rho2_sum
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_lr4_clip=10_log1p_max=100_rho2_sum


@REM lr3
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_lr3_clip=5e3_log1p_max=100_rho2_sum
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_lr3_clip=100_log1p_max=100_rho2_sum
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_lr3_clip=10_log1p_max=100_rho2_sum


@REM lr5e-5
call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --lr 5e-5 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_lr5e-5_clip=5e3_log1p_max=100_rho2_sum
call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --lr 5e-5 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_lr5e-5_clip=100_log1p_max=100_rho2_sum
call python train.py --model_class_name EcomDFCL_regretNet_tau --rho 0.01 --fcd_mode log1p --max_multiplier 100.0 --scheduler raw --lr 5e-5 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_lr5e-5_clip=10_log1p_max=100_rho2_sum




call python Evaluation.py
