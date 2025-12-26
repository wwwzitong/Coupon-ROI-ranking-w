
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.1 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=0.1_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.3 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=0.3_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.5 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=0.5_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 1.0 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=1.0_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 1.5 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=1.5_clip=5e3_boxcox
@REM call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 2.0 --clipnorm 5e3 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=2.0_clip=5e3_boxcox

call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.1 --clipnorm 100 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=0.1_clip=100_boxcox
call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.3 --clipnorm 100 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=0.3_clip=100_boxcox
call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 0.5 --clipnorm 100 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=0.5_clip=100_boxcox
call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 1.0 --clipnorm 100 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=1.0_clip=100_boxcox
call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 1.5 --clipnorm 100 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=1.5_clip=100_boxcox
call python train_boxcox.py --model_class_name EcomDFCL_v3 --alpha 2.0 --clipnorm 100 --model_path ./model/EcomDFCL_v3_2pll_2pos_gradient_lr3_alpha=2.0_clip=100_boxcox

call python Evaluation.py