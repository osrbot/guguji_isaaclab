# Design Notes

## Reward design

The locomotion reward balances forward progress with posture quality and gait regularity.

| Term | Weight | Purpose |
|---|---:|---|
| `track_lin_vel_x_exp` | +4.8 | Track commanded forward velocity |
| `forward_progress` | +6.0 | Reward actual forward motion |
| `alive_bonus` | +0.6 | Keep the episode alive |
| `upright` | +1.6 | Maintain torso stability |
| `height` | +0.9 | Hold a consistent base height |
| `hip_alternation` | +2.0 | Encourage left-right alternation |
| `knee_flexion` | +0.8 | Maintain meaningful knee lift |
| `feet_air_time` | +1.5 | Reward feet leaving the ground |
| `yaw_rate` | -0.5 | Penalize spinning |
| `lateral_velocity` | -0.3 | Penalize sideways drift |
| `knee_symmetry` | -1.0 | Penalize limping asymmetry |
| `backward_velocity` | -2.8 | Penalize backward motion |
| `stall_penalty` | -4.6 | Penalize standing still under a move command |
| `action_rate` | -0.004 | Smooth actions |
| `joint_pos_limits` | -0.05 | Stay within joint limits |
| `undesired_knee_contacts` | -1.0 | Discourage knee-ground collisions |

## Why use a reference gait

Instead of predicting full joint trajectories from scratch, the policy predicts **residual joint targets** on top of a sinusoidal reference gait.

This gives three benefits:

1. Lower exploration burden in early training.
2. More interpretable gait structure.
3. Faster convergence toward a usable walking policy.

## Reference gait parameters

| Parameter | Value | Meaning |
|---|---:|---|
| `gait_period` | 0.72 s | Full gait cycle duration |
| `stance_ratio` | 0.55 | Portion of time spent in stance |
| `hip_pitch_amplitude` | 0.45 rad | Hip swing magnitude |
| `knee_pitch_amplitude` | 0.60 rad | Knee flexion magnitude during swing |
| `swing_knee_scale` | 1.35 | Extra swing-phase knee lift |
| `scale` | 0.12 | Residual correction budget |

## Recent design iterations

### Fixing circling behavior

A policy can look like it is moving forward while actually rotating if the reward only sees body-frame x velocity. To prevent that, the environment now penalizes yaw rate and strengthens lateral-velocity punishment.

### Fixing limping and weak leg lift

Two issues showed up during iteration:

- one leg dominated the gait,
- the swing leg did not lift enough.

The response was to:

- increase `hip_alternation`,
- add explicit knee symmetry punishment,
- increase `feet_air_time`,
- and raise the reference gait amplitudes.

### Improving compatibility with newer `rsl-rl`

The project also updated actor distribution handling and export logic to stay compatible with `rsl-rl >= 5.0`, including migration away from deprecated stochastic configuration fields.

## Design philosophy

The repository aims for a practical middle ground:

- simple enough to inspect and tune,
- structured enough to scale beyond a toy example,
- and explicit enough that reward shaping decisions remain understandable.

!!! note
    The current design is intentionally task-focused. It prioritizes stable forward locomotion and clear gait structure over building a large multi-task benchmark surface.
