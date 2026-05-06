# Changelog Highlights

This page summarizes the most important recent changes. For the full history, see `CHANGELOG.md` in the repository root.

## v0.9.0 · 2026-05-06

### Reward hardening — sim-to-real robustness

Benchmarked against agibot_x1_train and unitree_rl_lab; six gaps were closed:

- **Velocity tracking moved to yaw frame** — body-frame x velocity tilts with the robot; yaw-frame stays horizontal regardless of roll/pitch.
- `joint_pos_limits` weight **-0.05 → -2.0** — previous weight was 40× weaker than comparable projects, allowing frequent limit collisions that damage real hardware.
- `action_rate` weight **-0.004 → -0.03** — reduces joint chatter and motor heating.
- Added `lin_vel_z` **(−1.5)** — penalizes vertical bouncing; all three reference projects include this.
- Added `ang_vel_xy` **(−0.1)** — penalizes roll/pitch angular velocity; previous design only penalized yaw.
- Added `feet_slide` **(−0.1)** — penalizes horizontal foot velocity during contact, closing a common sim-to-real gap.

## v0.8.0 · 2026-05-02

### README and docs refresh

- Bumped Isaac Sim to 5.1.0 and Isaac Lab to Latest in badges and requirements.
- Added a **Fine-tuning from a checkpoint** section covering `--resume True`, specific run/checkpoint loading, and cross-task transfer (flat → rough).
- Added a **Star History** chart to both READMEs.
- Mirrored all changes into the Chinese README and docs.

## v0.7.0 · 2026-04-24

### PPO training divergence fixes

Five root causes of training instability were identified and resolved:

- `init_std` 1.0 → **0.1** — old value caused ±0.08 rad per-step noise, short episodes, and std growing to 12+ during training
- `num_steps_per_env` 24 → **2048** — 24 steps (0.48s) was shorter than one gait cycle (0.72s); GAE could not see a complete stride
- `max_iterations` 500 → **30000** (flat), 1500 → **5000** (rough) — biped locomotion needs far more iterations than the template default
- `clip_actions` None → **10.0** — unbounded outputs with large std immediately toppled the robot
- `num_mini_batches` 4 → **32** — matches the larger rollout buffer
- `reset_robot_joints position_range` (0.85, 1.15) → **(1.0, 1.0)** — scale-based perturbation is meaningless for near-zero joints
- `RayCasterCfg`: `attach_yaw_only=True` → `ray_alignment="yaw"` (API update)

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
