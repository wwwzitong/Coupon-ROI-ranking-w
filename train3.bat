
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.1  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.1_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.3  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.3_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.5  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.5_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 1.0  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=1.0_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 1.5  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=1.5_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 2.0  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=2.0_clip=5e3_boxcox

@REM call python train_boxcox_copy.py --model_class_name EcomDFCL_v3 --alpha 0.1  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.1_clip=100_boxcox
@REM call python train_boxcox_copy.py --model_class_name EcomDFCL_v3 --alpha 0.3  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.3_clip=100_boxcox
@REM call python train_boxcox_copy.py --model_class_name EcomDFCL_v3 --alpha 0.5  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.5_clip=100_boxcox
@REM call python train_boxcox_copy.py --model_class_name EcomDFCL_v3 --alpha 1.0  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=1.0_clip=100_boxcox
call python train_boxcox_copy.py --model_class_name EcomDFCL_v3 --alpha 1.5  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=1.5_clip=100_boxcox
call python train_boxcox_copy.py --model_class_name EcomDFCL_v3 --alpha 2.0  --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=2.0_clip=100_boxcox

@REM call python train_fcd_copy.py --model_class_name EcomDFCL_v3 --alpha 0.1 --fcd_mode log1p --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.1_clip=100_global_log1p
@REM call python train_fcd_copy.py --model_class_name EcomDFCL_v3 --alpha 0.3 --fcd_mode log1p --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.3_clip=100_global_log1p
@REM call python train_fcd_copy.py --model_class_name EcomDFCL_v3 --alpha 0.5 --fcd_mode log1p --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr4_CD_alpha=0.5_clip=100_global_log1p


call python Evaluation3.py