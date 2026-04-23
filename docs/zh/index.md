# Guguji Isaac Lab 中文首页

<div class="hero-grid">
  <div>
    <span class="hero-kicker">Isaac Lab 扩展</span>
    <h1 class="hero-title">面向 Guguji 双足机器人的 GPU 并行运动控制训练</h1>
    <p class="hero-subtitle">
      这是一个面向 <strong>Guguji</strong> 双足机器人的 Isaac Lab 训练扩展，覆盖训练、评估与策略导出流程。
      项目将原有 Gazebo + ROS 2 + Stable-Baselines3 工作流迁移到 Isaac Lab + RSL-RL，以获得更高的仿真并行度和更快的策略迭代速度。
    </p>
    <div class="hero-actions">
      <a class="hero-button hero-button-primary" href="getting-started/">中文快速开始</a>
      <a class="hero-button" href="training/">中文训练说明</a>
      <a class="hero-button" href="../">English</a>
    </div>
  </div>
  <div class="hero-media">
    <img alt="Guguji 平地行走" src="https://raw.githubusercontent.com/osrbot/guguji_isaaclab/main/img/guguji_velocity_flat.gif">
  </div>
</div>

## 项目亮点

<div class="card-grid">
  <div class="feature-card">
    <h3>高并行训练</h3>
    <p>在 GPU 上同时运行数千个环境，相比单环境 Gazebo 流程大幅提高训练效率。</p>
  </div>
  <div class="feature-card">
    <h3>参考步态 + 残差策略</h3>
    <p>策略输出叠加在正弦参考步态上的残差关节目标，降低探索难度并加快收敛。</p>
  </div>
  <div class="feature-card">
    <h3>面向部署</h3>
    <p>评估流程支持导出 TorchScript 和 ONNX，方便接入后续部署链路。</p>
  </div>
</div>

## 环境一览

| Gym ID | 地形 | 环境数 | 用途 |
|---|---|---:|---|
| `Isaac-Velocity-Flat-Guguji-v0` | 平坦 | 4096 | 主训练入口 |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | 平坦 | 50 | 评估与可视化 |
| `Isaac-Velocity-Rough-Guguji-v0` | 粗糙 | 2048 | 泛化训练 |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | 粗糙 | 50 | 粗糙地形评估 |

## 推荐阅读路径

- 先看 [中文快速开始](getting-started.md)
- 再看 [中文训练与评估](training.md)
- 需要更完整技术说明时，可继续阅读英文版的 [Design Notes](../design/)
