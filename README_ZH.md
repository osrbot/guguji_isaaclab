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
        ├── velocity_env_cfg.py    # 基础环境配置
        ├── mdp/
        │   ├── rewards.py         # 自定义奖励函数
        │   └── curriculums.py     # 课程学习
        └── config/guguji/
            ├── __init__.py        # Gym 环境注册
            ├── flat_env_cfg.py    # 平坦地形配置
            ├── rough_env_cfg.py   # 粗糙地形配置
            └── agents/
                └── rsl_rl_ppo_cfg.py  # PPO 超参数
```

## 奖励函数说明

主要奖励项（来自 guguji_rl 迁移，权重已调优）：

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| 速度跟踪 | 4.8 | 指数核，σ=0.10 |
| 前向进度 | 6.0 | 正向位移奖励 |
| 直立保持 | 1.6 | `exp(-4*(roll²+pitch²))` |
| 高度保持 | 0.9 | `exp(-8*height_error²)` |
| 髋关节交替 | 0.5 | 左右髋反相运动 |
| 膝关节弯曲 | 0.35 | 目标弯曲角 0.28 rad |
| 存活奖励 | 0.6/步 | — |
| 跌倒惩罚 | -15.0 | 终止时触发 |

## 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。
