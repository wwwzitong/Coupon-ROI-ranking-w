python SDP_analysis.py \
    --model_path ../final-ECLIFT/model/EcomDFCL_regretNet_rplusc_wce_bs256_step500_lr1e-4_clip=5e3_max=1_tau=1.0_seed40 \
    --sample_npz sample_features_ECLIFT.npz \
    --lambda_cost 0.5 \
    --epsilon 0.1 \
    --emp_n 1000 --emp_eps 0.01 \
    --out_json sdp_lipschitz_report.json