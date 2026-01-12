
@REM v2_tau
@REM lr4
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr4_clip=5e3_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr4_clip=100_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr4_clip=10_log1p_max=1

@REM lr4+CD
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler warmup+decay --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr4_CD_clip=5e3_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler warmup+decay --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr4_CD_clip=100_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler warmup+decay --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr4_CD_clip=10_log1p_max=1

@REM lr3
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr3_clip=5e3_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr3_clip=100_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr3_clip=10_log1p_max=1

@REM lr3+CD
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler warmup+decay --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr3_CD_clip=5e3_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler warmup+decay --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr3_CD_clip=100_log1p_max=1
call python train.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --max_multiplier 1.0 --scheduler warmup+decay --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_wce_2pos_lr3_CD_clip=10_log1p_max=1
