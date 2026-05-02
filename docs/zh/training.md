# 中文训练与评估

## 平地训练

建议从这里开始。

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Guguji-v0 \
  --num_envs=4096 \
  --headless
```

平地任务使用前向速度课程，初始速度约为 **0.10 m/s**，会随着跟踪效果提升逐步扩展到 **0.30 m/s**。

## 粗糙地形训练

当平地策略稳定后，再切换到粗糙地形：

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Rough-Guguji-v0 \
  --num_envs=2048 \
  --headless
```

## 评估策略

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
  --num_envs=50
```

## 加载指定检查点

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
  --num_envs=50 \
  --load_run=2026-04-17_15-33-29 \
  --checkpoint=model_19999.pt
```

## 微调 / 继续训练

在已有良好模型的基础上继续训练——适用于中断后续训练、或在收敛策略上调整奖励权重后微调。

**从最新 run 的最新 checkpoint 续训：**

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Guguji-v0 \
  --num_envs=4096 \
  --headless \
  --resume True
```

**指定 run 和 checkpoint 文件续训：**

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Guguji-v0 \
  --num_envs=4096 \
  --headless \
  --load_run=2026-04-17_15-33-29 \
  --checkpoint=model_29999.pt
```

**平地策略迁移到粗糙地形（跨任务微调）：**

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
  --task=Isaac-Velocity-Rough-Guguji-v0 \
  --num_envs=2048 \
  --headless \
  --load_run=2026-04-17_15-33-29 \
  --checkpoint=model_29999.pt
```

!!! tip
    运行 `ls logs/rsl_rl/guguji_flat/` 可列出所有可用 run，找到要续训的时间戳。

## 输出产物

训练日志与模型文件位于：

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

评估脚本还会导出：

- `policy.pt`：TorchScript 部署文件
- `policy.onnx`：ONNX 部署文件

## TensorBoard

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs/rsl_rl/guguji_flat
```

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs/rsl_rl/guguji_rough
```

## 实践建议

1. 开始长时间训练前先验证环境注册是否成功
2. 优先把平地直行走稳
3. 重点检查是否存在绕圈、跛行、抬腿不足等现象
4. 在 `play.py` 中验证最好的一版策略后再导出
5. 最后再进入粗糙地形泛化训练
