#!/bin/bash

MODEL=EcomOneStepCCB

for lr in 1e-3; do
  for bs in 256 512; do
    for ips_clip in 5 10 20; do
      for entropy_coef in 1e-3 1e-2; do
        for tau in 0.5 1 2; do
          for clipnorm in 5e3; do

            model_path=./model/${MODEL}_bs${bs}_step500_lr${lr}_ipsclip${ips_clip}_ent${entropy_coef}_clipnorm${clipnorm}_tau${tau}

            python train.py \
              --model_class_name ${MODEL} \
              --tau ${tau} \
              --bs ${bs} \
              --lr ${lr} \
              --ips_clip ${ips_clip} \
              --entropy_coef ${entropy_coef} \
              --clipnorm ${clipnorm} \
              --model_path ${model_path}

          done
        done
      done
    done
  done
done

