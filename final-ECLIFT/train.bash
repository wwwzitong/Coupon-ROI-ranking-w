
# python train.py --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3

# python train.py --model_class_name EcomDFCL_v3 --alpha 10 --loss_function 2pll --bs 256 --lr 1e-3 --clipnorm 5e3 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10
# python train.py --model_class_name EcomDFCL_v3 --alpha 0.1 --loss_function 3erl --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5
# python train.py --model_class_name EcomDFCL_v3 --alpha 100 --loss_function 4ifdl --bs 512 --lr 1e-3 --clipnorm 100 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100

# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0

python train.py --model_class_name EcomOneStepCCB --tau 1.0 --bs 256 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomOneStepCCB_bs256_step500_lr1e-4_clip=5e3_tau=1.0
python train.py --model_class_name EcomOneStepCCB --tau 1.0 --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/EcomOneStepCCB_bs256_step500_lr1e-3_clip=5e3_tau=1.0
python train.py --model_class_name EcomOneStepCCB --tau 1.0 --bs 256 --lr 5e-3 --clipnorm 5e3 --model_path ./model/EcomOneStepCCB_bs256_step500_lr5e-3_clip=5e3_tau=1.0
python train.py --model_class_name EcomOneStepCCB --tau 1.0 --bs 256 --lr 5e-4 --clipnorm 5e3 --model_path ./model/EcomOneStepCCB_bs256_step500_lr5e-4_clip=5e3_tau=1.0
python train.py --model_class_name EcomOneStepCCB --tau 1.0 --bs 256 --lr 1e-5 --clipnorm 5e3 --model_path ./model/EcomOneStepCCB_bs256_step500_lr1e-5_clip=5e3_tau=1.0
python train.py --model_class_name EcomOneStepCCB --tau 1.0 --bs 256 --lr 5e-5 --clipnorm 5e3 --model_path ./model/EcomOneStepCCB_bs256_step500_lr5e-5_clip=5e3_tau=1.0

python Evaluation_CCB.py
# run2:

# python train.py --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3_run2

# python train.py --model_class_name EcomDFCL_v3 --alpha 10 --loss_function 2pll --bs 256 --lr 1e-3 --clipnorm 5e3 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10_run2
# python train.py --model_class_name EcomDFCL_v3 --alpha 0.1 --loss_function 3erl --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5_run2
# python train.py --model_class_name EcomDFCL_v3 --alpha 100 --loss_function 4ifdl --bs 512 --lr 1e-3 --clipnorm 100 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100_run2

# python train.py --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_run2




# seed 41:

# python train.py --seed 41 --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3_seed41

# python train.py --seed 41 --model_class_name EcomDFCL_v3 --alpha 10 --loss_function 2pll --bs 256 --lr 1e-3 --clipnorm 5e3 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10_seed41
# python train.py --seed 41 --model_class_name EcomDFCL_v3 --alpha 0.1 --loss_function 3erl --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5_seed41
# python train.py --seed 41 --model_class_name EcomDFCL_v3 --alpha 100 --loss_function 4ifdl --bs 512 --lr 1e-3 --clipnorm 100 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100_seed41

# python train.py --seed 41 --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed41


# seed 43:

# python train.py --seed 43 --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3_seed43

# python train.py --seed 43 --model_class_name EcomDFCL_v3 --alpha 10 --loss_function 2pll --bs 256 --lr 1e-3 --clipnorm 5e3 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10_seed43
# python train.py --seed 43 --model_class_name EcomDFCL_v3 --alpha 0.1 --loss_function 3erl --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5_seed43
# python train.py --seed 43 --model_class_name EcomDFCL_v3 --alpha 100 --loss_function 4ifdl --bs 512 --lr 1e-3 --clipnorm 100 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100_seed43

# python train.py --seed 43 --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed43


# seed 40:

# python train.py --seed 40 --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3_seed40

# python train.py --seed 40 --model_class_name EcomDFCL_v3 --alpha 10 --loss_function 2pll --bs 256 --lr 1e-3 --clipnorm 5e3 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10_seed40
# python train.py --seed 40 --model_class_name EcomDFCL_v3 --alpha 0.1 --loss_function 3erl --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5_seed40
# python train.py --seed 40 --model_class_name EcomDFCL_v3 --alpha 100 --loss_function 4ifdl --bs 512 --lr 1e-3 --clipnorm 100 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100_seed40

# python train.py --seed 40 --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40


# seed 39:

# python train.py --seed 39 --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3_seed39

# python train.py --seed 39 --model_class_name EcomDFCL_v3 --alpha 10 --loss_function 2pll --bs 256 --lr 1e-3 --clipnorm 5e3 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10_seed39
# python train.py --seed 39 --model_class_name EcomDFCL_v3 --alpha 0.1 --loss_function 3erl --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5_seed39
# python train.py --seed 39 --model_class_name EcomDFCL_v3 --alpha 100 --loss_function 4ifdl --bs 512 --lr 1e-3 --clipnorm 100 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100_seed39

# python train.py --seed 39 --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed39


# seed 44:

# python train.py --seed 44 --model_class_name SLearner --bs 256 --lr 1e-3 --clipnorm 5e3 --model_path ./model/SLearner_wce_mean_bs256_step500_lr1e-3_clip=5e3_seed44

# python train.py --seed 44 --model_class_name EcomDFCL_v3 --alpha 10 --loss_function 2pll --bs 256 --lr 1e-3 --clipnorm 5e3 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_2pll_bs256_step500_lr1e-3_clip=5e3_alpha=10_seed44
# python train.py --seed 44 --model_class_name EcomDFCL_v3 --alpha 0.1 --loss_function 3erl --bs 512 --lr 1e-3 --clipnorm 100 --tau 2.5 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_3erl_bs512_step500_lr1e-3_clip=100_alpha=0.1_tau=2.5_seed44
# python train.py --seed 44 --model_class_name EcomDFCL_v3 --alpha 100 --loss_function 4ifdl --bs 512 --lr 1e-3 --clipnorm 100 --batch_sum_mean sum --model_path ./model/EcomDFCL_v3_wce_4ifdl_bs512_step500_lr1e-3_clip=100_alpha=100_seed44

# python train.py --seed 44 --model_class_name EcomDFCL_regretNet_rplusc --tau 1.0 --bs 256 --max_multiplier 1.0 --lr 1e-4 --clipnorm 5e3 --model_path ./model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed44


python Evaluation_boundary.py

