#!/bin/bash

# loop over tasks
for task in "mujoco-reacher"; do
    CUDA_VISIBLE_DEVICES=7 python generate_demos.py enable_wandb=false task=$task +num_demos=20 env_mode=sync compile=false
done
