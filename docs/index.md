# Guguji Isaac Lab

<div class="hero-grid">
  <div>
    <span class="hero-kicker">Isaac Lab Extension</span>
    <h1 class="hero-title">Modern GPU-parallel locomotion training for the Guguji biped</h1>
    <p class="hero-subtitle">
      A focused Isaac Lab extension for training, evaluating, and exporting locomotion policies on the <strong>Guguji</strong> robot.
      The project migrates the workflow from Gazebo + ROS 2 + Stable-Baselines3 to Isaac Lab + RSL-RL for faster iteration and cleaner sim-to-real training loops.
    </p>
    <div class="hero-actions">
      <a class="hero-button hero-button-primary" href="getting-started/">Get started</a>
      <a class="hero-button" href="training/">Training guide</a>
      <a class="hero-button" href="https://github.com/osrbot/guguji_isaaclab">GitHub</a>
    </div>
    <div class="badge-row">
      <img alt="Isaac Sim 4.5.0" src="https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg">
      <img alt="Isaac Lab 2.1.0" src="https://img.shields.io/badge/IsaacLab-2.1.0-silver.svg">
      <img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg">
      <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-yellow.svg">
    </div>
  </div>
  <div class="hero-media">
    <img alt="Guguji flat terrain locomotion" src="https://raw.githubusercontent.com/osrbot/guguji_isaaclab/main/img/guguji_velocity_flat.gif">
  </div>
</div>

## Why this project

<div class="card-grid">
  <div class="feature-card">
    <h3>Parallel training</h3>
    <p>Run thousands of environments in parallel on GPU instead of iterating one-by-one in a traditional Gazebo setup.</p>
  </div>
  <div class="feature-card">
    <h3>Residual gait policy</h3>
    <p>The policy predicts residual joint positions on top of a sinusoidal reference gait, reducing exploration burden and improving convergence speed.</p>
  </div>
  <div class="feature-card">
    <h3>Built for deployment</h3>
    <p>Evaluation exports TorchScript and ONNX artifacts so the trained policy can move cleanly into downstream deployment workflows.</p>
  </div>
</div>

## At a glance

| Gym ID | Terrain | Envs | Use |
|---|---|---:|---|
| `Isaac-Velocity-Flat-Guguji-v0` | Flat | 4096 | Main training entry point |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | Flat | 50 | Evaluation and visualization |
| `Isaac-Velocity-Rough-Guguji-v0` | Rough | 2048 | Terrain generalization training |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | Rough | 50 | Rough-terrain evaluation |

## Migration context

| | Legacy setup | This repository |
|---|---|---|
| Simulator | Gazebo Fortress | Isaac Sim / PhysX 5 |
| RL stack | Stable-Baselines3 | RSL-RL |
| Parallel environments | 1 | 4096 |
| Curriculum | Manual multi-stage script | Native curriculum manager |
| Export | `.zip` | `.pt` TorchScript / `.onnx` |

## Quick start

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m pip install -e ~/Desktop/guguji_simulation/guguji_isaaclab/source/guguji_locomotion
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/list_envs.py
```

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Guguji-v0 \
  --num_envs=4096 \
  --headless
```

## What you will find in the docs

- **Getting Started**: installation, environment setup, and repository layout.
- **Training & Evaluation**: standard train / play / export workflows.
- **Design Notes**: reward shaping, reference gait design, and curriculum highlights.
- **Changelog**: recent fixes and gait-quality improvements.

## Repository structure

```text
guguji_isaaclab/
├── scripts/
│   ├── list_envs.py
│   └── rsl_rl/
│       ├── train.py
│       ├── play.py
│       └── cli_args.py
└── source/guguji_locomotion/
    └── guguji_locomotion/
        ├── assets/
        └── tasks/locomotion/velocity/
            ├── velocity_env_cfg.py
            ├── mdp/
            └── config/guguji/
```
