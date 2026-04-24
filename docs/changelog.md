# Changelog Highlights

This page summarizes the most important recent changes. For the full history, see `CHANGELOG.md` in the repository root.

## v0.6.0 · 2026-04-24

### Housekeeping and docs refresh

This release cleaned up leftover template code and updated the documentation:

- fixed `list_envs.py` table display — class entry points now render as `module.ClassName` with capped column widths,
- removed the unused `anymal_d` task config (template scaffold, never used),
- bumped Isaac Sim requirement to 5.1.0 and Isaac Lab to Latest,
- added Feishu internal setup guide link to getting-started,
- added a new Tested Environments page documenting the validated hardware and software stack.

## v0.5.0 · 2026-04-19

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

## v0.4.0 · 2026-04-18

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

## v0.3.0 · 2026-04-18

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

## v0.2.0 · 2026-04-16

### Core Guguji locomotion stack landed

Added:

- flat and rough environment configs,
- PPO configs,
- Gym registrations,
- reference gait action,
- gait phase observations,
- velocity command curriculum.

## v0.1.0 · 2026-04-16

### Repository bootstrap

Initial repository setup included:

- Isaac Lab extension initialization,
- English and Chinese readmes,
- migration context documentation,
- root changelog.
