
@REM v1_rc
@REM lr3
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_CD_clip=5e3_max=1
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_CD_clip=100_max=1
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr3_CD_clip=10_max=1

@REM @REM lr4
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_CD_clip=5e3_max=1
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_CD_clip=100_max=1
@REM call python train.py --model_class_name EcomDFCL_regretNet_rc --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rc_2pll_2pos_lr4_CD_clip=10_max=1




@REM v2_tau
@REM lr3_CD
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler warmup+decay --max_multiplier 100.0 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_CD_clip=5e3_max=100
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler warmup+decay --max_multiplier 100.0 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_CD_clip=100_max=100
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler warmup+decay --max_multiplier 100.0 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_CD_clip=10_max=100

@REM @REM lr4_CD
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler warmup+decay --max_multiplier 100.0 --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_CD_clip=5e3_max=100
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler warmup+decay --max_multiplier 100.0 --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_CD_clip=100_max=100
@REM call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler warmup+decay --max_multiplier 100.0 --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_CD_clip=10_max=100

@REM lr3
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 100.0 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=5e3_max=100
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 100.0 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=100_max=100
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 100.0 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=10_max=100

@REM lr4
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 100.0 --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=5e3_max=100
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 100.0 --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=100_max=100
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 100.0 --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=10_max=100

@REM lr3 10
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 10.0 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=5e3_max=10
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 10.0 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=100_max=10
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 10.0 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr3_clip=10_max=10

@REM lr4 10
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 10.0 --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=5e3_max=10
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 10.0 --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=100_max=10
call python train.py --model_class_name EcomDFCL_regretNet_tau --scheduler raw --max_multiplier 10.0 --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_tau_2pll_2pos_lr4_clip=10_max=10


@REM v1_rplusc
@REM lr3+CD
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler warmup+decay --max_multiplier 10.0 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_CD_clip=5e3_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler warmup+decay --max_multiplier 10.0 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_CD_clip=100_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler warmup+decay --max_multiplier 10.0 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_CD_clip=10_max=10

@REM lr4+CD
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler warmup+decay --max_multiplier 10.0 --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_CD_clip=5e3_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler warmup+decay --max_multiplier 10.0 --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_CD_clip=100_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler warmup+decay --max_multiplier 10.0 --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_CD_clip=10_max=10

@REM lr3
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler raw --max_multiplier 10.0 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=5e3_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler raw --max_multiplier 10.0 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=100_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler raw --max_multiplier 10.0 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr3_clip=10_max=10

@REM lr4
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler raw --max_multiplier 10.0 --lr 0.0001 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=5e3_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler raw --max_multiplier 10.0 --lr 0.0001 --clipnorm 100 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=100_max=10
call python train.py --model_class_name EcomDFCL_regretNet_rplusc --scheduler raw --max_multiplier 10.0 --lr 0.0001 --clipnorm 10 --model_path ./model/EcomDFCL_regretNet_rplusc_2pll_2pos_lr4_clip=10_max=10

@REM call python Evaluation.py
