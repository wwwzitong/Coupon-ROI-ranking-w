
python train.py --alpha 1.0 --loss_function 2pll --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3

python Evaluation.py

