
@REM v1_rc
@REM lr3
call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_clip=5e3_raw_max=1
call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode raw --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_clip=100_raw_max=1
call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode raw --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_clip=10_raw_max=1

call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode log1p --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_clip=5e3_log1p_max=1
call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_clip=100_log1p_max=1
call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode log1p --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_clip=10_log1p_max=1

@REM lr4
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode raw --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_clip=5e3_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode raw --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_clip=100_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode raw --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_clip=10_raw_max=1

@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode log1p --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_clip=5e3_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode log1p --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_clip=100_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rc --fcd_mode log1p --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_clip=10_log1p_max=1



@REM @REM v1_rplusc
@REM @REM lr3
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=5e3_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode raw --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=100_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode raw --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=10_raw_max=1

@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode log1p --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=5e3_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=100_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode log1p --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=10_log1p_max=1

@REM @REM lr4
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode raw --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=5e3_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode raw --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=100_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode raw --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=10_raw_max=1

@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode log1p --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=5e3_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode log1p --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=100_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_rplusc --fcd_mode log1p --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=10_log1p_max=1



@REM @REM v2_tau
@REM @REM lr3
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode raw --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=5e3_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode raw --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=100_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode raw --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=10_raw_max=1

@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=5e3_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=100_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=10_log1p_max=1

@REM @REM lr4
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode raw --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=5e3_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode raw --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=100_raw_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode raw --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=10_raw_max=1

@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=5e3_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=100_log1p_max=1
@REM call python train_fcd.py --model_class_name EcomDFCL_regretNet_tau --fcd_mode log1p --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=10_log1p_max=1


@REM call python Evaluation.py