#!/bin/bash

# loop over tasks

for task in "ms-pick-hammer" "ms-pick-spatula" "ms-pick-scissors"; do
    CUDA_VISIBLE_DEVICES=3 python train.py task=$task exp_name=0716-default
done
