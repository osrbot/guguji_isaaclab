# 测试环境

本页记录已验证可正常运行本项目的硬件与软件配置。

## 硬件

| 组件 | 规格 |
|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS (Noble Numbat) x86_64 |
| 内核 | Linux 6.17.0-22-generic |
| CPU | AMD Ryzen 9 7950X3D（32 线程）@ 5.76 GHz |
| GPU（训练） | NVIDIA GeForce RTX 4080 SUPER |
| GPU（集成） | AMD Raphael |
| 内存 | 32 GB DDR5 |

## 软件

| 组件 | 版本 |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | Latest (main) |
| RSL-RL | >= 5.0 |
| Python | 3.10 |
| ROS 2 | Jazzy |
| CUDA | 12.x（随 Isaac Sim 捆绑） |

## 仿真环境

以下四个 Gym 环境均已在上述配置上完成验证：

| 环境 ID | 地形 | 用途 |
|---|---|---|
| `Isaac-Velocity-Flat-Guguji-v0` | 平地 | 训练 |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | 平地 | 评估 / 可视化 |
| `Isaac-Velocity-Rough-Guguji-v0` | 粗糙地形 | 训练 |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | 粗糙地形 | 评估 / 可视化 |

## 注意事项

!!! note
    Isaac Sim 5.1.0 自带 Python 3.10 解释器和 CUDA 运行时，请勿将系统 Python 与 Isaac Lab 环境混用。

!!! tip
    在 RTX 4080 SUPER 上以 2048 个并行环境训练，平地任务约可达 10–15k FPS；粗糙地形因高度扫描计算略慢。
