#!/bin/bash

# loop over tasks

# ogbench
# 'og-ant', 'og-antball', 'og-point-arena', 'og-point-maze', 'og-point-bottleneck',
# 'og-point-circle', 'og-point-spiral', 'og-ant-arena', 'og-ant-maze', 'og-ant-bottleneck',
# 'og-ant-circle', 'og-ant-spiral',

# pygame
# 'pygame-cowboy', 'pygame-coinrun', 'pygame-spaceship', 'pygame-pong', 'pygame-bird-attack',
# 'pygame-highway', 'pygame-landing', 'pygame-air-hockey', 'pygame-rocket-collect', 'pygame-chase-evade',
# 'pygame-coconut-dodge', 'pygame-cartpole-balance', 'pygame-cartpole-swingup', 'pygame-cartpole-balance-sparse', 'pygame-cartpole-swingup-sparse',
# 'pygame-cartpole-tremor', 'pygame-point-maze-var1', 'pygame-point-maze-var2', 'pygame-point-maze-var3',

# atari
# 'atari-alien', 'atari-assault', 'atari-asterix', 'atari-atlantis', 'atari-bank-heist',
# 'atari-battle-zone', 'atari-beamrider', 'atari-boxing', 'atari-chopper-command', 'atari-crazy-climber',
# 'atari-double-dunk', 'atari-gopher', 'atari-ice-hockey', 'atari-jamesbond', 'atari-kangaroo',
# 'atari-krull', 'atari-ms-pacman', 'atari-name-this-game', 'atari-phoenix', 'atari-pong',
# 'atari-road-runner', 'atari-robotank', 'atari-seaquest', 'atari-space-invaders', 'atari-tutankham',
# 'atari-upndown', 'atari-yars-revenge',

# ogbench
for task in "og-ant" "og-antball" "og-point-arena" "og-point-maze" "og-point-bottleneck" "og-point-circle" "og-point-spiral" "og-ant-arena" "og-ant-maze" "og-ant-bottleneck" "og-ant-circle" "og-ant-spiral"; do
    CUDA_VISIBLE_DEVICES=2 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done

# pygame
for task in "pygame-cowboy" "pygame-coinrun" "pygame-spaceship" "pygame-pong" "pygame-bird-attack" "pygame-highway" "pygame-landing" "pygame-air-hockey" "pygame-rocket-collect" "pygame-chase-evade" "pygame-coconut-dodge" "pygame-cartpole-balance" "pygame-cartpole-swingup" "pygame-cartpole-balance-sparse" "pygame-cartpole-swingup-sparse" "pygame-cartpole-tremor" "pygame-point-maze-var1" "pygame-point-maze-var2" "pygame-point-maze-var3"; do
    CUDA_VISIBLE_DEVICES=2 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done

# atari
for task in "atari-alien" "atari-assault" "atari-asterix" "atari-atlantis" "atari-bank-heist" "atari-battle-zone" "atari-beamrider" "atari-boxing" "atari-chopper-command" "atari-crazy-climber" "atari-double-dunk" "atari-gopher" "atari-ice-hockey" "atari-jamesbond" "atari-kangaroo" "atari-krull" "atari-ms-pacman" "atari-name-this-game" "atari-phoenix" "atari-pong" "atari-road-runner" "atari-robotank" "atari-seaquest" "atari-space-invaders" "atari-tutankham" "atari-upndown" "atari-yars-revenge"; do
    CUDA_VISIBLE_DEVICES=2 python train.py exp_name=0914-single-bc task=$task seed=1 bc_baseline=true compile=false
done
