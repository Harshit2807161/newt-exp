#!/bin/bash

# loop over tasks
# "cartpole-balance-long-sparse" "cartpole-balance-two-poles-sparse" "walker-stand-incline" "walker-walk-incline" "walker-run-incline" "spinner-jump-four" "ms-push-apple" "ms-push-pear" "ms-push-can" "ms-push-sponge" "ms-push-banana" "ms-push-screwdriver" "ms-pick-rubiks-cube" "ms-pick-cup" "ms-pick-golf-ball" "ms-pick-soccer-ball" "og-point-var1" "pygame-point-maze-var4" "pygame-reacher-easy" "pygame-reacher-hard"


for task in "ms-pick-rubiks-cube" "ms-pick-cup" "ms-pick-golf-ball" "ms-pick-soccer-ball"; do
    CUDA_VISIBLE_DEVICES=1 python train.py task=$task seed=1 finetune=true checkpoint=/data/nihansen/code/tdmpc25/checkpoints/soup-0901-default-100M.pt exp_name=0923-ft-new-instr utd=0.5 num_envs=100 lr_schedule=warmup
done

# tasks as comma-separated list
# cartpole-balance-long-sparse,cartpole-balance-two-poles-sparse,walker-stand-incline,walker-walk-incline,walker-run-incline,spinner-jump-four,ms-push-apple,ms-push-pear,ms-push-can,ms-push-sponge,ms-push-banana,ms-push-screwdriver,ms-pick-rubiks-cube,ms-pick-cup,ms-pick-golf-ball,ms-pick-soccer-ball,og-point-var1,pygame-point-maze-var4,pygame-reacher-easy,pygame-reacher-hard