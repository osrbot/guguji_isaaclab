# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.2.0] - 2026-04-16

### Added
- **Step 5** — Guguji-specific flat/rough env configs (`config/guguji/flat_env_cfg.py`, `rough_env_cfg.py`)
  - `GugujiRoughEnvCfg` / `GugujiRoughEnvCfg_PLAY`: wires `GUGUJI_CFG`, 2048 envs, reference gait action, gait phase obs
  - `GugujiFlatEnvCfg` / `GugujiFlatEnvCfg_PLAY`: flat plane, no height scan, velocity curriculum starting at 0.18 m/s
- **Step 5** — Guguji PPO agent configs (`config/guguji/agents/rsl_rl_ppo_cfg.py`)
  - `GugujiFlatPPORunnerCfg`: net=[256,256,128], lr=6e-5, clip=0.15, ent_coef=0.0 (matches `walk_ppo.yaml`)
  - `GugujiRoughPPORunnerCfg`: net=[512,256,128], 1500 iterations
- **Step 6** — Gym environment registration (`config/guguji/__init__.py`)
  - `Isaac-Velocity-Flat-Guguji-v0`, `Isaac-Velocity-Flat-Guguji-Play-v0`
  - `Isaac-Velocity-Rough-Guguji-v0`, `Isaac-Velocity-Rough-Guguji-Play-v0`
  - Wired into import chain: `velocity/__init__.py` → `config/__init__.py` → `config/guguji/__init__.py`
- **Step 7** — Reference gait `ActionTerm` (`mdp/actions.py`)
  - `ReferenceGaitAction` / `ReferenceGaitActionCfg`: vectorised PyTorch port of the Gazebo sinusoidal gait
  - Per-env phase tracking; left/right legs 180° out of phase; resets on episode end
  - Policy outputs residual joint positions added on top of the reference trajectory
- **Step 7** — Gait phase observation (`mdp/observations.py`)
  - `gait_phase_obs`: exposes `(sin, cos)` of current gait phase to the policy
- **Step 8** — Velocity command curriculum (`mdp/curriculums.py`)
  - `velocity_command_curriculum`: continuously advances command velocity upper bound (0.18 → 0.26 m/s) as tracking success improves, mirroring the 3-stage `walk_ppo.yaml` curriculum

### Changed
- `mdp/__init__.py`: exports `ReferenceGaitAction`, `ReferenceGaitActionCfg`, `gait_phase_obs`, `velocity_command_curriculum`
- `velocity/__init__.py`: imports `config` sub-package to trigger Gym registrations on package load
- `config/__init__.py`: imports `guguji` sub-package

---

## [0.1.0] - 2026-04-16

### Added
- Initialized repository from `IsaacLabExtensionTemplate`, renamed to `guguji_locomotion`
- `README.md` and `README_ZH.md` with project overview and migration context
- `CHANGELOG.md`
- Parent repo `.gitignore` updated to exclude `guguji_isaaclab/` (independent git sub-project)

### Migration Context
- Source project: `guguji_rl` (Stable-Baselines3 + Gazebo Fortress + ROS 2 Jazzy)
- Target: Isaac Lab 2.1.0 + RSL-RL + Isaac Sim 4.5.0
- Robot: Guguji biped, 8 joints (hip/knee/ankle_pitch + ankle per leg)
- Trained policies in source: balance (200k steps) + walk curriculum (3 stages, up to 0.26 m/s)
