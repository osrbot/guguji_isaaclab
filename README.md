# Guguji Locomotion — Isaac Lab Extension

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

Isaac Lab extension for training locomotion policies on the **Guguji** biped robot.
Migrated from a Gazebo/ROS 2 + Stable-Baselines3 setup to Isaac Lab + RSL-RL for GPU-accelerated parallel training.

For the Chinese guide, see [README_ZH.md](README_ZH.md).

## Overview

This extension provides:

- `Guguji-Isaac-Velocity-Flat-v0` — flat terrain velocity-tracking task
- `Guguji-Isaac-Velocity-Rough-v0` — rough terrain with height scanner and terrain curriculum

**Key differences from the original Gazebo setup:**

| | Gazebo (guguji_rl) | Isaac Lab (this repo) |
|---|---|---|
| Parallel envs | 1 | 4096 |
| Simulator | Gazebo Fortress | Isaac Sim / PhysX 5 |
| RL framework | Stable-Baselines3 | RSL-RL |
| Policy export | `.zip` | `.pt` TorchScript / `.onnx` |

## Installation

Install Isaac Lab following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then:

```bash
python -m pip install -e source/guguji_locomotion
```

Verify:

```bash
python scripts/list_envs.py
```

## Training

```bash
# Flat terrain (recommended for initial tuning)
python scripts/rsl_rl/train.py --task=Guguji-Isaac-Velocity-Flat-v0 --num_envs=4096 --headless

# Rough terrain
python scripts/rsl_rl/train.py --task=Guguji-Isaac-Velocity-Rough-v0 --num_envs=4096 --headless
```

## Evaluation

```bash
python scripts/rsl_rl/play.py --task=Guguji-Isaac-Velocity-Flat-Play-v0 --num_envs=50
```

## Project Structure

```
source/guguji_locomotion/
└── guguji_locomotion/
    ├── assets/              # Robot ArticulationCfg
    └── tasks/locomotion/velocity/
        ├── velocity_env_cfg.py   # Base env config
        ├── mdp/
        │   ├── rewards.py        # Custom reward functions
        │   └── curriculums.py
        └── config/guguji/
            ├── flat_env_cfg.py
            ├── rough_env_cfg.py
            └── agents/
                └── rsl_rl_ppo_cfg.py
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
