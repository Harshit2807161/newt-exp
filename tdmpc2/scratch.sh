#!/bin/bash

# loop over tasks

for task in "walker-stand-incline" "walker-walk-incline" "walker-run-incline" "walker-arabesque" "walker-legs-up" "walker-headstand" "spinner-spin-four" "spinner-spin-backward-four" "spinner-jump-four"; do
    CUDA_VISIBLE_DEVICES=0 python train.py task=$task utd=0.5 num_envs=1 compile=false exp_name=0906-scratch
done
