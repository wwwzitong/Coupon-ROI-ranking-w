
# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-4 --loss_function 4ifdl --alpha 1 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-4_alpha=1_clip=100_log1p
# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 5e-4 --loss_function 4ifdl --alpha 1 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr5e-4_alpha=1_clip=100_log1p
# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 100 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=100_clip=100_log1p
# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 2e-3 --loss_function 4ifdl --alpha 1 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr2e-3_alpha=1_clip=100_log1p

# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-4 --loss_function 4ifdl --alpha 1 --fcd_mode log1p --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-4_alpha=1_clip=5e3_log1p
# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 5e-4 --loss_function 4ifdl --alpha 1 --fcd_mode log1p --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr5e-4_alpha=1_clip=5e3_log1p
# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 100 --fcd_mode log1p --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=100_clip=5e3_log1p
# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 2e-3 --loss_function 4ifdl --alpha 1 --fcd_mode log1p --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr2e-3_alpha=1_clip=5e3_log1p

# python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 1 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=1_clip=100_log1p
python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 0.1 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=0.1_clip=100_log1p
python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 0.5 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=0.5_clip=100_log1p
python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 1.5 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=1.5_clip=100_log1p
python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 10 --fcd_mode log1p --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=10_clip=100_log1p

python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 0.1 --fcd_mode raw --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=0.1_clip=100_raw
python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 0.5 --fcd_mode raw --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=0.5_clip=100_raw
python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 1.5 --fcd_mode raw --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=1.5_clip=100_raw
python train_fcd.py --model_class_name EcomDFCL_v3 --lr 1e-3 --loss_function 4ifdl --alpha 10 --fcd_mode raw --clipnorm 100 --model_path ./model/EcomDFCL_v3_4ifdl_ratios_bs256_2pos_lr1e-3_alpha=10_clip=100_raw

python Evaluation.py