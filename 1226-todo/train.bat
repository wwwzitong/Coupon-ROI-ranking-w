@REM @REM lr3
@REM call python train.py --model_class_name SLearner --clipnorm 5e3 --model_path ./model/SLearner_2pos_lr3_clip=5e3

@REM call python train.py --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100

@REM call python train.py --model_class_name SLearner --clipnorm 10 --model_path ./model/SLearner_2pos_lr3_clip=10



@REM @REM @REM lr4
@REM call python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 5e3 --model_path ./model/SLearner_2pos_lr4_clip=5e3

@REM call python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 100 --model_path ./model/SLearner_2pos_lr4_clip=100

@REM call python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 10 --model_path ./model/SLearner_2pos_lr4_clip=10



@REM @REM lr4+CD
@REM call python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 5e3 --model_path ./model/SLearner_2pos_lr4_CD_clip=5e3

@REM call python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 100 --model_path ./model/SLearner_2pos_lr4_CD_clip=100

call python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 10 --model_path ./model/SLearner_2pos_lr4_CD_clip=10



@REM lr3+CD
call python train.py --model_class_name SLearner --lr 0.001 --clipnorm 5e3 --model_path ./model/SLearner_2pos_lr3_CD_clip=5e3

call python train.py --model_class_name SLearner --lr 0.001 --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_CD_clip=100

call python train.py --model_class_name SLearner --lr 0.001 --clipnorm 10 --model_path ./model/SLearner_2pos_lr3_CD_clip=10

@REM call python Evaluation.py