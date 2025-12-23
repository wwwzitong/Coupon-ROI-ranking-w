call python train_copy.py --model_class_name GLU_base_DFCL --alpha 0.1 --fcd_mode raw --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=0.1_clip=100_global_raw
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 0.3 --fcd_mode raw --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=0.3_clip=100_global_raw
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 0.5 --fcd_mode raw --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=0.5_clip=100_global_raw
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 1.0 --fcd_mode raw --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=1.0_clip=100_global_raw
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 1.5 --fcd_mode raw --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=1.5_clip=100_global_raw
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 2.0 --fcd_mode raw --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=2.0_clip=100_global_raw

call python train_copy.py --model_class_name GLU_base_DFCL --alpha 0.1 --fcd_mode log1p --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=0.1_clip=100_global_log1p
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 0.3 --fcd_mode log1p --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=0.3_clip=100_global_log1p
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 0.5 --fcd_mode log1p --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=0.5_clip=100_global_log1p
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 1.0 --fcd_mode log1p --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=1.0_clip=100_global_log1p
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 1.5 --fcd_mode log1p --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=1.5_clip=100_global_log1p
call python train_copy.py --model_class_name GLU_base_DFCL --alpha 2.0 --fcd_mode log1p --model_path ./model/GLU_base_DFCL_2pll_2pos_gradient_lr4_CD_alpha=2.0_clip=100_global_log1p


call python Evaluation2.py