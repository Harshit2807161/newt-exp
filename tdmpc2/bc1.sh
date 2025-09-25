#!/bin/bash

# loop over tasks

# dmcontrol + dmcontrol ext
for task in "walker-stand" "walker-walk" "walker-run" "cheetah-run" "reacher-easy" "reacher-hard" "acrobot-swingup" "pendulum-swingup" "cartpole-balance" "cartpole-balance-sparse" "cartpole-swingup" "cartpole-swingup-sparse" "cup-catch" "finger-spin" "finger-turn-easy" "finger-turn-hard" "fish-swim" "hopper-stand" "hopper-hop" "quadruped-walk" "quadruped-run" "walker-walk-backward" "walker-run-backward" "cheetah-run-backward" "cheetah-run-front" "cheetah-run-back" "cheetah-jump" "hopper-hop-backward" "reacher-three-easy" "reacher-three-hard" "cup-spin" "pendulum-spin" "jumper-jump" "spinner-spin" "spinner-spin-backward" "spinner-jump" "giraffe-run"; do
    CUDA_VISIBLE_DEVICES=2 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done

# metaworld
for task in "mw-assembly" "mw-basketball" "mw-button-press-topdown" "mw-button-press-topdown-wall" "mw-button-press" "mw-button-press-wall" "mw-coffee-button" "mw-coffee-pull" "mw-coffee-push" "mw-dial-turn" "mw-disassemble" "mw-door-open" "mw-door-close" "mw-drawer-close" "mw-drawer-open" "mw-faucet-open" "mw-faucet-close" "mw-hammer" "mw-handle-press-side" "mw-handle-press" "mw-handle-pull-side" "mw-handle-pull" "mw-lever-pull" "mw-peg-insert-side" "mw-peg-unplug-side" "mw-pick-out-of-hole" "mw-pick-place" "mw-pick-place-wall" "mw-plate-slide" "mw-plate-slide-side" "mw-plate-slide-back" "mw-plate-slide-back-side" "mw-push-back" "mw-push" "mw-push-wall" "mw-reach" "mw-reach-wall" "mw-soccer" "mw-stick-push" "mw-stick-pull"  "mw-sweep-into"  "mw-sweep"  "mw-window-open"  "mw-window-close"  "mw-bin-picking"  "mw-box-close"  "mw-door-lock"  "mw-door-unlock"  "mw-hand-insert"; do
    CUDA_VISIBLE_DEVICES=2 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done
