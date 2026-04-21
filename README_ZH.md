# Guguji 运动控制 — Isaac Lab 扩展

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

基于 Isaac Lab 的 **Guguji 双足机器人**运动控制策略训练扩展。
从 Gazebo/ROS 2 + Stable-Baselines3 迁移而来，利用 Isaac Lab 的 GPU 并行仿真大幅提升训练速度。

英文说明见 [README.md](README.md)。迁移详细流程见 [../docs/isaaclab_migration_guide.md](../docs/isaaclab_migration_guide.md)。

## 概述

本扩展提供以下训练环境：

| Gym ID | 地形 | 环境数 | 用途 |
|--------|------|--------|------|
| `Isaac-Velocity-Flat-Guguji-v0` | 平坦 | 4096 | 训练（速度课程 0.10→0.30 m/s） |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | 平坦 | 50 | 评估 / 可视化 |
| `Isaac-Velocity-Rough-Guguji-v0` | 粗糙 | 2048 | 训练（地形课程） |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | 粗糙 | 50 | 评估 / 可视化 |

**与原 Gazebo 方案的关键差异：**

| 方面 | Gazebo（guguji_rl） | Isaac Lab（本仓库） |
|------|--------------------|--------------------|
| 并行环境数 | 1 | 4096 |
| 仿真器 | Gazebo Fortress | Isaac Sim / PhysX 5 |
| 训练框架 | Stable-Baselines3 | RSL-RL |
| 策略导出 | `.zip` | `.pt` TorchScript / `.onnx` |
| 课程学习 | 手动多阶段脚本 | 原生 CurriculumManager |

## 环境要求

- Isaac Lab 2.1.0 + Isaac Sim 4.5.0
- `rsl-rl-lib >= 5.0`（随 Isaac Lab 安装包附带）
- Python 3.10

## 安装

按照 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab，然后安装本扩展：

```bash
cd ~/rlgpu_ws/IsaacLab
python -m pip install -e ~/Desktop/guguji_simulation/guguji_isaaclab/source/guguji_locomotion
```

验证环境注册是否成功：

```bash
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/list_envs.py
```

## 训练

所有训练命令均需在 Isaac Lab 根目录下通过 `isaaclab.sh -p` 运行：

```bash
cd ~/rlgpu_ws/IsaacLab

# 平坦地形训练（推荐从这里开始）
# 速度课程：从 0.10 m/s 起步，随跟踪成功率提升逐步增加到 0.30 m/s
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Guguji-v0 \
    --num_envs=4096 \
    --headless

# 粗糙地形训练（平坦地形策略收敛后进行）
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Guguji-v0 \
    --num_envs=2048 \
    --headless
```

训练日志和模型检查点保存在 `logs/rsl_rl/<experiment_name>/<timestamp>/` 目录下。

## 评估

```bash
cd ~/rlgpu_ws/IsaacLab

# 平坦地形评估（自动加载最新检查点）
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
    --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
    --num_envs=50

# 指定特定检查点
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/rsl_rl/play.py \
    --task=Isaac-Velocity-Flat-Guguji-Play-v0 \
    --num_envs=50 \
    --load_run=2026-04-17_15-33-29 \
    --checkpoint=model_19999.pt
```

play 脚本运行结束后会自动将策略导出到 `logs/.../exported/policy.pt`（TorchScript）和 `policy.onnx`，可直接用于部署。


## 分析

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs/rsl_rl/guguji_flat
```

## 项目结构

```
guguji_isaaclab/
├── scripts/
│   ├── list_envs.py               # 验证环境注册
│   └── rsl_rl/
│       ├── train.py               # 训练入口
│       ├── play.py                # 评估 / 策略导出入口
│       └── cli_args.py            # 共用 CLI 参数工具
└── source/guguji_locomotion/
    └── guguji_locomotion/
        ├── assets/
        │   └── guguji.py          # GUGUJI_CFG — 从 URDF 生成的 ArticulationCfg
        └── tasks/locomotion/velocity/
            ├── velocity_env_cfg.py    # 基础环境配置（场景、观测、奖励、终止条件）
            ├── mdp/
            │   ├── actions.py         # ReferenceGaitAction：正弦参考步态 + 残差策略
            │   ├── observations.py    # gait_phase_obs：(sin, cos) 相位供策略条件化
            │   ├── rewards.py         # 自定义奖励函数（前进、直立、步态质量）
            │   └── curriculums.py     # velocity_command_curriculum
            └── config/guguji/
                ├── __init__.py            # Gym 环境注册
                ├── flat_env_cfg.py        # 平坦地形 + 速度课程
                ├── rough_env_cfg.py       # 粗糙地形 + 参考步态参数
                └── agents/
                    └── rsl_rl_ppo_cfg.py  # PPO 超参数
```

## 奖励设计

奖励函数在前进运动和步态质量之间取得平衡：

| 奖励项 | 权重 | 作用 |
|--------|------|------|
| `track_lin_vel_x_exp` | +4.8 | 跟踪指令前进速度 |
| `forward_progress` | +6.0 | 奖励实际前进位移 |
| `alive_bonus` | +0.6 | 存活奖励 |
| `upright` | +1.6 | 保持躯干直立 |
| `height` | +0.9 | 维持目标躯干高度（0.32 m） |
| `hip_alternation` | +2.0 | 左右髋关节反相运动（交替迈步） |
| `knee_flexion` | +0.8 | 维持目标膝关节弯曲角度（0.38 rad） |
| `feet_air_time` | +1.5 | 每只脚都有离地时间 |
| `yaw_rate` | -0.5 | 惩罚偏航旋转（防止绕圈） |
| `lateral_velocity` | -0.3 | 惩罚侧向漂移 |
| `knee_symmetry` | -1.0 | 惩罚左右膝关节不对称（防止跛行） |
| `backward_velocity` | -2.8 | 惩罚后退 |
| `stall_penalty` | -4.6 | 有指令时惩罚原地不动 |
| `action_rate` | -0.004 | 平滑动作输出 |
| `joint_pos_limits` | -0.05 | 保持在关节限位内 |
| `undesired_knee_contacts` | -1.0 | 惩罚膝盖触地 |

## 参考步态

策略输出的是叠加在正弦参考轨迹上的**残差关节位置**，大幅缩小了探索空间，加速学习收敛。

当前参考步态关键参数（`rough_env_cfg.py`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `gait_period` | 0.72 s | 完整步态周期时长 |
| `stance_ratio` | 0.55 | 支撑相占比 |
| `hip_pitch_amplitude` | 0.45 rad | 髋关节摆动峰值角度 |
| `knee_pitch_amplitude` | 0.60 rad | 摆动相膝关节弯曲峰值 |
| `swing_knee_scale` | 1.35 | 摆动相膝关节额外缩放系数 |
| `scale`（残差幅度） | 0.12 | 策略残差预算 |

## 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。


[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

基于 Isaac Lab 的 **Guguji 双足机器人**运动控制策略训练扩展。
从 Gazebo/ROS 2 + Stable-Baselines3 迁移而来，利用 Isaac Lab 的 GPU 并行仿真大幅提升训练速度。

英文说明见 [README.md](README.md)。迁移详细流程见 [../docs/isaaclab_migration_guide.md](../docs/isaaclab_migration_guide.md)。

## 概述

本扩展提供以下训练环境：

- `Guguji-Isaac-Velocity-Flat-v0` — 平坦地形速度跟踪任务
- `Guguji-Isaac-Velocity-Flat-Play-v0` — 平坦地形推理/评估（50个环境，无噪声）
- `Guguji-Isaac-Velocity-Rough-v0` — 粗糙地形 + 高度扫描器 + 地形课程
- `Guguji-Isaac-Velocity-Rough-Play-v0` — 粗糙地形推理/评估

**与原 Gazebo 方案的关键差异：**

| 方面 | Gazebo（guguji_rl） | Isaac Lab（本仓库） |
|------|--------------------|--------------------|
| 并行环境数 | 1 | 4096 |
| 仿真器 | Gazebo Fortress | Isaac Sim / PhysX 5 |
| 训练框架 | Stable-Baselines3 | RSL-RL |
| 策略导出 | `.zip` | `.pt` TorchScript / `.onnx` |
| 课程学习 | 手动多阶段脚本 | 原生 CurriculumManager |

## 安装

按照 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab，然后：

```bash
python -m pip install -e source/guguji_locomotion
```

验证安装：

```bash
python scripts/list_envs.py
```

## 训练

```bash
# 平坦地形（推荐用于初期调参）
python scripts/rsl_rl/train.py --task=Guguji-Isaac-Velocity-Flat-v0 --num_envs=4096 --headless

# 粗糙地形
python scripts/rsl_rl/train.py --task=Guguji-Isaac-Velocity-Rough-v0 --num_envs=4096 --headless
```

## 评估

```bash
python scripts/rsl_rl/play.py --task=Guguji-Isaac-Velocity-Flat-Play-v0 --num_envs=50
```

## 项目结构

```
source/guguji_locomotion/
└── guguji_locomotion/
    ├── assets/                    # 机器人 ArticulationCfg 定义
    └── tasks/locomotion/velocity/
        ├── velocity_env_cfg.py    # 基础环境配置（场景、观测、奖励、终止条件）
        ├── mdp/
        │   ├── actions.py         # ReferenceGaitAction — 正弦参考步态 + 残差策略
        │   ├── observations.py    # gait_phase_obs（sin/cos 相位，供策略条件化）
        │   ├── rewards.py         # 自定义奖励函数（从 guguji_rl 迁移）
        │   └── curriculums.py     # terrain_levels_vel + velocity_command_curriculum
        └── config/guguji/
            ├── __init__.py            # Gym 环境注册
            ├── flat_env_cfg.py        # 平坦地形 + 速度课程（0.18→0.26 m/s）
            ├── rough_env_cfg.py       # 粗糙地形 + 高度扫描器 + 地形课程
            └── agents/
                └── rsl_rl_ppo_cfg.py  # PPO 超参数（Flat + Rough 两个变体）
```

## 已注册环境

| Gym ID | 地形 | 环境数 | 用途 |
|--------|------|--------|------|
| `Isaac-Velocity-Flat-Guguji-v0` | 平坦 | 4096 | 训练（速度课程 0.18→0.26 m/s） |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | 平坦 | 50 | 评估 / 可视化 |
| `Isaac-Velocity-Rough-Guguji-v0` | 粗糙 | 2048 | 训练（地形课程） |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | 粗糙 | 50 | 评估 / 可视化 |

## 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。
