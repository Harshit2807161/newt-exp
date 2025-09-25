#!/bin/bash

# loop over tasks

for task in "pygame-point-maze-var8"; do
    CUDA_VISIBLE_DEVICES=7 python train.py task=$task finetune=true checkpoint=/data/nihansen/code/tdmpc25/checkpoints/soup-0828-state-L-priorcoef10-100M.pt exp_name=0906-ft utd=0.5 num_envs=1 lr_schedule=warmup
done
