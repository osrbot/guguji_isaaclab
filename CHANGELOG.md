# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### In Progress
- Step 2: Define Guguji robot `ArticulationCfg` asset
- Step 3: Create base `LocomotionVelocityEnvCfg`
- Step 4: Port reward functions to Isaac Lab MDP style
- Step 5: Create Guguji-specific flat/rough env configs and PPO agent config
- Step 6: Register Gym environments
- Step 7: Port reference gait as custom `ActionTerm`
- Step 8: Migrate curriculum learning

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
