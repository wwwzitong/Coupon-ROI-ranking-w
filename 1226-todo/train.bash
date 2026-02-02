lr3
python train.py --model_class_name SLearner --clipnorm 5e3 --model_path ./model/SLearner_2pos_lr3_clip=5e3

python train.py --model_class_name SLearner --clipnorm 100 --model_path ./model/SLearner_2pos_lr3_clip=100

python train.py --model_class_name SLearner --clipnorm 10 --model_path ./model/SLearner_2pos_lr3_clip=10



# lr4
# python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 5e3 --model_path ./model/SLearner_2pos_lr4_clip=5e3

# python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 100 --model_path ./model/SLearner_2pos_lr4_clip=100

# python train.py --model_class_name SLearner --lr 0.0001 --clipnorm 10 --model_path ./model/SLearner_2pos_lr4_clip=10


python Evaluation.py