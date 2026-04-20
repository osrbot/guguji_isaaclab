# Guguji Locomotion — Isaac Lab Extension

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

Isaac Lab extension for training locomotion policies on the **Guguji** biped robot.
Migrated from a Gazebo/ROS 2 + Stable-Baselines3 setup to Isaac Lab + RSL-RL for GPU-accelerated parallel training.

For the Chinese guide, see [README_ZH.md](README_ZH.md).

## Overview

This extension provides four registered Gym environments:

| Gym ID | Terrain | Envs | Use |
|--------|---------|------|-----|
| `Isaac-Velocity-Flat-Guguji-v0` | Flat | 4096 | Training (velocity curriculum 0.10→0.30 m/s) |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | Flat | 50 | Evaluation / visualization |
| `Isaac-Velocity-Rough-Guguji-v0` | Rough | 2048 | Training (terrain curriculum) |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | Rough | 50 | Evaluation / visualization |

**Key differences from the original Gazebo setup:**

| | Gazebo (guguji_rl) | Isaac Lab (this repo) |
|---|---|---|
| Parallel envs | 1 | 4096 |
| Simulator | Gazebo Fortress | Isaac Sim / PhysX 5 |
| RL framework | Stable-Baselines3 | RSL-RL |
| Policy export | `.zip` | `.pt` TorchScript / `.onnx` |
| Curriculum | Manual multi-stage script | Native CurriculumManager |

## Requirements

- Isaac Lab 2.1.0 with Isaac Sim 4.5.0
- `rsl-rl-lib >= 5.0` (bundled with the Isaac Lab installation above)
- Python 3.10

## Installation

Install Isaac Lab following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then install this extension:

```bash
cd ~/rlgpu_ws/IsaacLab
python -m pip install -e ~/Desktop/guguji_simulation/guguji_isaaclab/source/guguji_locomotion
```

Verify the environments are registered:

```bash
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/list_envs.py
```

## Training

All training commands should be run from the Isaac Lab root directory using `isaaclab.sh -p`:

```bash
cd ~/rlgpu_ws/IsaacLab

# Flat terrain — recommended starting point
# Velocity curriculum: starts at 0.10 m/s, advances to 0.30 m/s as tracking improves
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Guguji-v0 \
    --num_envs=4096 \
    --headless

# Rough terrain — after flat terrain policy converges
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Guguji-v0 \
    --num_envs=2048 \
    --headless
```

Training logs and checkpoints are saved to `logs/rsl_rl/<experiment_name>/<timestamp>/`.

## Evaluation

```bash
cd ~/rlgpu_ws/IsaacLab

# Flat terrain evaluation (loads latest checkpoint automatically)
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
    --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
    --num_envs=50

# Load a specific checkpoint
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
    --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
    --num_envs=50 \
    --load_run=2026-04-17_15-33-29 \
    --checkpoint=model_19999.pt
```

The play script also exports the policy to `logs/.../exported/policy.pt` (TorchScript) and `policy.onnx` for deployment.

## Project Structure

```
guguji_isaaclab/
├── scripts/
│   ├── list_envs.py               # Verify environment registration
│   └── rsl_rl/
│       ├── train.py               # Training entry point
│       ├── play.py                # Evaluation / policy export entry point
│       └── cli_args.py            # Shared CLI argument helpers
└── source/guguji_locomotion/
    └── guguji_locomotion/
        ├── assets/
        │   └── guguji.py          # GUGUJI_CFG — ArticulationCfg from URDF
        └── tasks/locomotion/velocity/
            ├── velocity_env_cfg.py    # Base env (scene, obs, rewards, terminations)
            ├── mdp/
            │   ├── actions.py         # ReferenceGaitAction: sinusoidal gait + residual
            │   ├── observations.py    # gait_phase_obs: (sin, cos) phase for policy
            │   ├── rewards.py         # Custom rewards (forward, upright, gait quality)
            │   └── curriculums.py     # velocity_command_curriculum
            └── config/guguji/
                ├── __init__.py            # Gym environment registrations
                ├── flat_env_cfg.py        # Flat terrain + velocity curriculum
                ├── rough_env_cfg.py       # Rough terrain + reference gait params
                └── agents/
                    └── rsl_rl_ppo_cfg.py  # PPO hyperparameters
```

## Reward Design

The reward function balances forward locomotion with gait quality:

| Term | Weight | Purpose |
|------|--------|---------|
| `track_lin_vel_x_exp` | +4.8 | Track commanded forward velocity |
| `forward_progress` | +6.0 | Reward actual forward displacement |
| `alive_bonus` | +0.6 | Survive the episode |
| `upright` | +1.6 | Keep base upright |
| `height` | +0.9 | Maintain target base height (0.32 m) |
| `hip_alternation` | +2.0 | Left/right hips in anti-phase (stride alternation) |
| `knee_flexion` | +0.8 | Maintain target knee flexion (0.38 rad) |
| `feet_air_time` | +1.5 | Each foot spends time in the air |
| `yaw_rate` | -0.5 | Penalize spinning / circling |
| `lateral_velocity` | -0.3 | Penalize sideways drift |
| `knee_symmetry` | -1.0 | Penalize left/right knee asymmetry |
| `backward_velocity` | -2.8 | Penalize moving backward |
| `stall_penalty` | -4.6 | Penalize standing still when commanded to move |
| `action_rate` | -0.004 | Smooth actions |
| `joint_pos_limits` | -0.05 | Stay within joint limits |
| `undesired_knee_contacts` | -1.0 | Penalize knee hitting the ground |

## Reference Gait

The policy outputs **residual joint positions** on top of a sinusoidal reference trajectory. This significantly reduces the exploration space and accelerates learning.

Key reference gait parameters (in `rough_env_cfg.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gait_period` | 0.72 s | Full gait cycle duration |
| `stance_ratio` | 0.55 | Fraction of cycle in stance |
| `hip_pitch_amplitude` | 0.45 rad | Peak hip swing angle |
| `knee_pitch_amplitude` | 0.60 rad | Peak knee flexion in swing |
| `swing_knee_scale` | 1.35 | Extra knee flexion multiplier during swing |
| `scale` (residual) | 0.12 | Policy residual budget |

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

