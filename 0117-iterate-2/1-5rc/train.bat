
@REM v1_rc
@REM lr4
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.8
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.0
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.5
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.0
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.5


@REM lr3
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.8
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.0
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.5
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.0
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.5


@REM @REM lr5e-5
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.8
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.0
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.5
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.0
call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.5



@REM @REM lr4 rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.8_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.5_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.5_rho2


@REM @REM lr3 rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.8_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.5_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.5_rho2


@REM @REM lr5e-5 rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.8_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.5_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.0_rho2
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.5_rho2



@REM @REM lr4 rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.8_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.5_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.5_rho3


@REM @REM lr3 rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.8_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.5_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.5_rho3


@REM @REM lr5e-5 rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.8_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.5_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.0_rho3
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.5_rho3



@REM @REM lr4 rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.8_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=1.5_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=2.5_rho4


@REM @REM lr3 rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.8_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=1.5_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=2.5_rho4


@REM @REM lr5e-5 rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.8_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=1.5_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.0_rho4
call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=2.5_rho4



@REM @REM add rho1
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.5
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.5
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.5

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=3.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=3.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=3.0

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=5.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=5.0
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=5.0


@REM @REM rho2
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.5_rho2
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.5_rho2
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.5_rho2

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=3.0_rho2
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=3.0_rho2
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=3.0_rho2

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=5.0_rho2
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=5.0_rho2
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.01 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=5.0_rho2


@REM @REM rho3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.5_rho3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.5_rho3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.5_rho3

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=3.0_rho3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=3.0_rho3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=3.0_rho3

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=5.0_rho3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=5.0_rho3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.001 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=5.0_rho3


@REM @REM rho4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=0.5_rho4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=0.5_rho4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 0.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=0.5_rho4

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=3.0_rho4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=3.0_rho4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 3.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=3.0_rho4

@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.0001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr4_clip=60_log1p_max=1.5+1_tau=5.0_rho4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 0.001 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr3_clip=60_log1p_max=1.5+1_tau=5.0_rho4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --rho 0.0001 --tau 5.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 5e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr5e-5_clip=60_log1p_max=1.5+1_tau=5.0_rho4





@REM @REM lr2e-5
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 2e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr2e-5_clip=60_log1p_max=1.5+1_tau=0.8
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 2e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr2e-5_clip=60_log1p_max=1.5+1_tau=1.0
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 2e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr2e-5_clip=60_log1p_max=1.5+1_tau=1.5
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 2e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr2e-5_clip=60_log1p_max=1.5+1_tau=2.0
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 2e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr2e-5_clip=60_log1p_max=1.5+1_tau=2.5


@REM @REM lr1e-5
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 0.8 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 1e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr1e-5_clip=60_log1p_max=1.5+1_tau=0.8
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 1e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr1e-5_clip=60_log1p_max=1.5+1_tau=1.0
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 1e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr1e-5_clip=60_log1p_max=1.5+1_tau=1.5
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.0 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 1e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr1e-5_clip=60_log1p_max=1.5+1_tau=2.0
@REM @REM call python train.py --model_class_name EcomDFCL_regretNet_rc --tau 2.5 --fcd_mode log1p --max_multiplier_paid 1.5 --max_multiplier_cost 1.0 --lr 1e-5 --clipnorm 60 --model_path ./model/EcomDFCL_regretNet_rc_wce_2pos_lr1e-5_clip=60_log1p_max=1.5+1_tau=2.5


@REM call python Evaluation.py
