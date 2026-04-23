# Changelog Highlights

This page summarizes the most important recent changes. For the full history, see `CHANGELOG.md` in the repository root.

## 0.5.0 · 2026-04-19

### Gait quality improvements

This release focused on two visible issues:

- the robot taking tiny, unclear steps,
- insufficient knee lift during swing.

Key updates included:

- larger hip swing amplitude,
- larger knee swing amplitude,
- stronger swing-phase knee scaling,
- shorter stance ratio,
- larger residual action budget.

It also increased reward targets for hip alternation and knee flexion, while making air-time rewards easier to trigger.

## 0.4.0 · 2026-04-18

### Stability and symmetry fixes

This release fixed:

- training crashes caused by invalid standard deviation handling,
- asymmetric limping behavior,
- weak alternation incentives.

Key updates included:

- migration to `distribution_cfg` with `std_type="log"`,
- stronger `hip_alternation` and `knee_flexion` rewards,
- a new knee symmetry penalty,
- stronger feet air-time encouragement.

## 0.3.0 · 2026-04-18

### `play.py` compatibility and anti-circling reward updates

This release addressed:

- `rsl_rl >= 5.0` compatibility in `play.py`,
- export path changes for TorchScript and ONNX,
- observation API updates,
- circling behavior caused by missing yaw-rate punishment.

It also:

- lowered the curriculum starting speed,
- increased the max curriculum speed,
- increased training iterations,
- added a small entropy bonus.

## 0.2.0 · 2026-04-16

### Core Guguji locomotion stack landed

Added:

- flat and rough environment configs,
- PPO configs,
- Gym registrations,
- reference gait action,
- gait phase observations,
- velocity command curriculum.

## 0.1.0 · 2026-04-16

### Repository bootstrap

Initial repository setup included:

- Isaac Lab extension initialization,
- English and Chinese readmes,
- migration context documentation,
- root changelog.
