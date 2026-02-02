


python train.py --model_class_name EcomDFCL_regretNet_rc --tau 1.0 --fcd_mode log1p --max_multiplier_paid 0.05 --max_multiplier_cost 0.4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rc_wce_lr3_clip=60_max=0.05+0.4_tau=1.0



python Evaluation.py
