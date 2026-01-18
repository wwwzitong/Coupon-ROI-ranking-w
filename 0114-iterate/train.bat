
@REM v1_rplusc
@REM lr4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_lr4_clip=5e3_log1p_max=1_tau=1.0_res
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_lr4_clip=100_log1p_max=1_tau=1.0_res
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_lr4_clip=10_log1p_max=1_tau=1.0_res

@REM lr3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_lr3_clip=5e3_log1p_max=1_tau=1.0_res
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_lr3_clip=100_log1p_max=1_tau=1.0_res
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_lr3_clip=10_log1p_max=1_tau=1.0_res

@REM lr5e-5
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 5e-5 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_lr5e-5_clip=5e3_log1p_max=1_tau=1.0_res
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 5e-5 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_lr5e-5_clip=100_log1p_max=1_tau=1.0_res
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --scheduler raw --lr 5e-5 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_lr5e-5_clip=10_log1p_max=1_tau=1.0_res


call python Evaluation.py
