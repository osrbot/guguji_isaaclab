# Design Notes

## Reward design

The locomotion reward balances forward progress with posture quality and gait regularity.

| Term | Weight | Purpose |
|---|---:|---|
| `track_lin_vel_x_exp` | +4.8 | Track commanded forward velocity (yaw frame) |
| `forward_progress` | +6.0 | Reward actual forward motion (yaw frame) |
| `alive_bonus` | +0.6 | Keep the episode alive |
| `upright` | +1.6 | Maintain torso stability |
| `height` | +0.9 | Hold a consistent base height |
| `hip_alternation` | +2.0 | Encourage left-right alternation |
| `knee_flexion` | +0.8 | Maintain meaningful knee lift |
| `feet_air_time` | +1.5 | Reward feet leaving the ground |
| `yaw_rate` | -0.5 | Penalize spinning |
| `ang_vel_xy` | -0.1 | Penalize roll/pitch angular velocity |
| `lin_vel_z` | -1.5 | Penalize vertical bouncing |
| `lateral_velocity` | -0.3 | Penalize sideways drift |
| `knee_symmetry` | -2.0 | Penalize limping asymmetry |
| `hip_symmetry` | -1.5 | Penalize non-antiphase hip motion |
| `backward_velocity` | -2.8 | Penalize backward motion (yaw frame) |
| `stall_penalty` | -4.6 | Penalize standing still under a move command (yaw frame) |
| `action_rate` | -0.03 | Smooth actions |
| `joint_pos_limits` | -2.0 | Stay within joint limits |
| `undesired_knee_contacts` | -1.0 | Discourage knee-ground collisions |
| `feet_slide` | -0.1 | Penalize feet sliding during contact |

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

### Hardening reward design against sim-to-real failure modes

Benchmarking against agibot_x1_train and unitree_rl_lab revealed several gaps:

- **Velocity tracking moved to yaw frame.** Body-frame x velocity tilts with the robot, producing incorrect reward signals when the robot leans. Yaw-frame velocity removes roll/pitch from the rotation before computing the error, keeping the signal horizontal regardless of tilt.
- **`joint_pos_limits` raised from -0.05 to -2.0.** The original weight was 40× weaker than comparable projects. Hardware joints hitting their limits is a common cause of actuator damage on real robots.
- **`lin_vel_z` added at -1.5.** Without this, the policy can learn to bounce rather than walk. All three reference projects include this term.
- **`ang_vel_xy` added at -0.1.** Penalizes roll and pitch angular velocity. The previous design only penalized yaw, allowing large fore-aft and lateral body sway.
- **`action_rate` raised from -0.004 to -0.03.** Reduces joint chatter, which causes motor heating on real hardware.
- **`feet_slide` added at -0.1.** Penalizes horizontal foot velocity during ground contact, closing a common sim-to-real gap where the policy learns to drag feet instead of lifting them.

### Improving compatibility with newer `rsl-rl`

The project also updated actor distribution handling and export logic to stay compatible with `rsl-rl >= 5.0`, including migration away from deprecated stochastic configuration fields.

## Design philosophy

The repository aims for a practical middle ground:

- simple enough to inspect and tune,
- structured enough to scale beyond a toy example,
- and explicit enough that reward shaping decisions remain understandable.

!!! note
    The current design is intentionally task-focused. It prioritizes stable forward locomotion and clear gait structure over building a large multi-task benchmark surface.
