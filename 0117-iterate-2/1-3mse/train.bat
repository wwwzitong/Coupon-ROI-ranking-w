
@REM v1_rplusc
@REM lr4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=0.8
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.0
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.5
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.0
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.5


@REM lr3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=0.8
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.0
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.5
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.0
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.5


@REM lr5e-5
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=0.8
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.0
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.5
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.0
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.5



@REM lr4 rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=0.8_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.5_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.5_rho2


@REM lr3 rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=0.8_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.5_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.5_rho2


@REM lr5e-5 rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=0.8_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.5_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.01 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.5_rho2



@REM lr4 rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=0.8_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.5_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.5_rho3


@REM lr3 rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=0.8_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.5_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.5_rho3


@REM lr5e-5 rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=0.8_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.5_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.001 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.5_rho3



@REM lr4 rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=0.8_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=1.5_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 0.0001 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr4_clip=80_log1p_max=1_tau=2.5_rho4


@REM lr3 rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=0.8_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=1.5_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr3_clip=80_log1p_max=1_tau=2.5_rho4


@REM lr5e-5 rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=0.8_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=1.5_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --rho 0.0001 --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 5e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr5e-5_clip=80_log1p_max=1_tau=2.5_rho4




@REM lr2e-5
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 2e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr2e-5_clip=80_log1p_max=1_tau=0.8
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 2e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr2e-5_clip=80_log1p_max=1_tau=1.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 2e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr2e-5_clip=80_log1p_max=1_tau=1.5
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 2e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr2e-5_clip=80_log1p_max=1_tau=2.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 2e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr2e-5_clip=80_log1p_max=1_tau=2.5


@REM lr1e-5
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 0.8 --fcd_mode log1p --max_multiplier 1.0 --lr 1e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr1e-5_clip=80_log1p_max=1_tau=0.8
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --fcd_mode log1p --max_multiplier 1.0 --lr 1e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr1e-5_clip=80_log1p_max=1_tau=1.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.5 --fcd_mode log1p --max_multiplier 1.0 --lr 1e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr1e-5_clip=80_log1p_max=1_tau=1.5
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.0 --fcd_mode log1p --max_multiplier 1.0 --lr 1e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr1e-5_clip=80_log1p_max=1_tau=2.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 2.5 --fcd_mode log1p --max_multiplier 1.0 --lr 1e-5 --clipnorm 80 --model_path ./model/EcomDFCL_regretNet_rplusc_mse_2pos_lr1e-5_clip=80_log1p_max=1_tau=2.5


@REM call python Evaluation.py
