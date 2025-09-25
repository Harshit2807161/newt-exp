#!/bin/bash

# loop over tasks

# maniskill
# 'ms-ant-walk', 'ms-ant-run', 'ms-cartpole-balance', 'ms-cartpole-swingup', 'ms-hopper-stand',
# 'ms-hopper-hop', 'ms-pick-cube', 'ms-pick-cube-eepose', 'ms-pick-cube-so', 'ms-poke-cube',
# 'ms-push-cube', 'ms-pull-cube', 'ms-pull-cube-tool', 'ms-stack-cube', 'ms-place-sphere',
# 'ms-lift-peg', 'ms-pick-apple', 'ms-pick-banana', 'ms-pick-can', 'ms-pick-hammer',
# 'ms-pick-fork', 'ms-pick-knife', 'ms-pick-mug', 'ms-pick-orange', 'ms-pick-screwdriver',
# 'ms-pick-spoon', 'ms-pick-tennis-ball', 'ms-pick-baseball', 'ms-pick-cube-xarm6', 'ms-pick-sponge',
# 'ms-anymal-reach', 'ms-reach', 'ms-reach-eepose', 'ms-reach-xarm6', 'ms-cartpole-balance-sparse',
# 'ms-cartpole-swingup-sparse',

# mujoco
# 'mujoco-ant', 'mujoco-halfcheetah', 'mujoco-hopper', 'mujoco-inverted-pendulum', 'mujoco-reacher',
# 'mujoco-walker',

# box2d
# 'bipedal-walker-flat', 'bipedal-walker-uneven', 'bipedal-walker-rugged', 'bipedal-walker-hills', 'bipedal-walker-obstacles',
# 'lunarlander-land', 'lunarlander-hover', 'lunarlander-takeoff', 'lunarlander-rough', 'lunarlander-crash', 'lunarlander-obstacles',

# robodesk
# 'rd-push-red', 'rd-push-green', 'rd-push-blue', 'rd-open-slide', 'rd-open-drawer',
# 'rd-flat-block-in-bin',

# maniskill
for task in "ms-ant-walk" "ms-ant-run" "ms-cartpole-balance" "ms-cartpole-swingup" "ms-hopper-stand" "ms-hopper-hop" "ms-pick-cube" "ms-pick-cube-eepose" "ms-pick-cube-so" "ms-poke-cube" "ms-push-cube" "ms-pull-cube" "ms-pull-cube-tool" "ms-stack-cube" "ms-place-sphere" "ms-lift-peg" "ms-pick-apple" "ms-pick-banana" "ms-pick-can" "ms-pick-hammer" "ms-pick-fork" "ms-pick-knife" "ms-pick-mug" "ms-pick-orange" "ms-pick-screwdriver" "ms-pick-spoon" "ms-pick-tennis-ball" "ms-pick-baseball" "ms-pick-cube-xarm6" "ms-pick-sponge" "ms-anymal-reach" "ms-reach" "ms-reach-eepose" "ms-reach-xarm6" "ms-cartpole-balance-sparse" "ms-cartpole-swingup-sparse"; do
    CUDA_VISIBLE_DEVICES=3 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done

# mujoco
for task in "mujoco-ant" "mujoco-halfcheetah" "mujoco-hopper" "mujoco-inverted-pendulum" "mujoco-reacher" "mujoco-walker"; do
    CUDA_VISIBLE_DEVICES=3 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done

# box2d
for task in "bipedal-walker-flat" "bipedal-walker-uneven" "bipedal-walker-rugged" "bipedal-walker-hills" "bipedal-walker-obstacles" "lunarlander-land" "lunarlander-hover" "lunarlander-takeoff" "lunarlander-rough" "lunarlander-crash" "lunarlander-obstacles"; do
    CUDA_VISIBLE_DEVICES=3 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done

# robodesk
for task in "rd-push-red" "rd-push-green" "rd-push-blue" "rd-open-slide" "rd-open-drawer" "rd-flat-block-in-bin"; do
    CUDA_VISIBLE_DEVICES=3 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done
