# Training & Evaluation

## Train on flat terrain

This is the recommended starting point.

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Guguji-v0 \
  --num_envs=4096 \
  --headless
```

The flat task uses a forward-velocity curriculum that begins at **0.10 m/s** and scales toward **0.30 m/s** as tracking improves.

## Train on rough terrain

After the flat policy becomes stable, move to the rough-terrain task.

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Rough-Guguji-v0 \
  --num_envs=2048 \
  --headless
```

## Evaluate a trained policy

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
  --num_envs=50
```

## Load a specific checkpoint

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
  --num_envs=50 \
  --load_run=2026-04-17_15-33-29 \
  --checkpoint=model_19999.pt
```

## Logging and artifacts

Training logs and checkpoints are written to:

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

The evaluation script also exports:

- `policy.pt` for TorchScript deployment
- `policy.onnx` for ONNX-based downstream use

## Inspect training with TensorBoard

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs/rsl_rl/guguji_flat
```

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs/rsl_rl/guguji_rough
```

## Practical workflow

1. Verify environment registration before starting long runs.
2. Train the flat policy first.
3. Check whether the robot walks straight instead of circling.
4. Export the best checkpoint and validate it with `play.py`.
5. Then move to rough terrain and iterate on reward shaping only if needed.

## Common training focus areas

### 1. Straight walking

The project explicitly penalizes yaw rate and lateral drift to avoid policies that appear to move forward in body frame while actually circling.

### 2. Gait symmetry

Recent reward shaping emphasizes left-right alternation and knee symmetry so the policy does not collapse into a limping pattern.

### 3. Higher-quality swing phase

Reference gait amplitudes and swing knee scaling were increased to encourage clearer leg lift and more visible step alternation.

!!! tip
    If early policies wobble or circle, inspect yaw-rate punishment, lateral drift punishment, and whether the reference gait amplitudes are strong enough to bias the motion in the right direction.
