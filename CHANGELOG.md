# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.8.0] - 2026-05-02

### Docs

- README.md / README_ZH.md: bumped Isaac Sim badge `4.5.0 → 5.1.0`, Isaac Lab `2.1.0 → Latest`.
- Added **Fine-tuning from a checkpoint** section (EN + ZH) to both READMEs and `docs/training.md` / `docs/zh/training.md`, covering three scenarios:
  - `--resume True` — resume from the latest checkpoint in the latest run
  - `--load_run` + `--checkpoint` — resume from a specific run and file
  - cross-task transfer (flat → rough terrain)
- Added **Star History** chart (star-history.com SVG API) to both READMEs.

---

## [0.7.0] - 2026-04-24

### Fixed — PPO training divergence and rollout issues

Five root causes of training instability were identified and resolved:

| # | Parameter | Before | After | Reason |
|---|---|---|---|---|
| 1 | `init_std` | 1.0 | **0.1** | std=1.0 × scale=0.08 → ±0.08 rad noise per step; episodes too short for useful gradients; std grew to 12+ |
| 2 | `num_steps_per_env` | 24 | **2048** | 24 × 0.02s = 0.48s < one gait cycle (0.72s); GAE could not see a complete stride |
| 3 | `max_iterations` (flat) | 500 | **30000** | Biped locomotion requires far more iterations than the template default |
| 4 | `max_iterations` (rough) | 1500 | **5000** | Same reason |
| 5 | `clip_actions` | None | **10.0** | Unbounded outputs with large std immediately toppled the robot |
| 6 | `num_mini_batches` | 4 | **32** | Matches the larger rollout buffer (2048 envs × 2048 steps) |
| 7 | `reset_robot_joints position_range` | (0.85, 1.15) | **(1.0, 1.0)** | Scale-based perturbation is meaningless for near-zero joints; adds noise in early training |

Also updated `RayCasterCfg` from deprecated `attach_yaw_only=True` to `ray_alignment="yaw"`.

---

## [0.6.0] - 2026-04-24

### Fixed

- `scripts/list_envs.py`: table columns were unreadable — `env_cfg_entry_point` printed as `<class '...'>` and entry-point strings were truncated arbitrarily. Added `_fmt()` helper to render class objects as `module.ClassName`, set `max_width=40` on Entry Point and Config columns, and updated table title to match Guguji branding.

### Removed

- `config/anymal_d/`: leftover scaffold from `IsaacLabExtensionTemplate` (flat/rough env configs, PPO config, `__init__.py`). This project targets only the Guguji biped; the directory was dead code.

### Docs

- Updated Isaac Sim requirement `4.5.0 → 5.1.0` and Isaac Lab to `Latest` in both EN and ZH getting-started pages.
- Added internal Feishu Isaac Lab setup guide link: <https://osrbotai.feishu.cn/wiki/QDj5w31Ynil8rYkyBNUc6tIVnwg>
- Added `docs/tested-environments.md` (EN) and `docs/zh/tested-environments.md` (ZH) documenting the validated hardware/software stack and all four Guguji Gym environments.
- Registered **Tested Environments** page in `mkdocs.yml` nav.

---

## [0.5.0] - 2026-04-19

### Changed — 步态质量优化（膝盖抬高 + 交替迈步）

本次改动针对机器人出现的"小碎步"和"膝盖抬起不足"问题，从参考轨迹幅度和奖励目标两个层面同时调整。

#### 参考步态幅度（`rough_env_cfg.py`）

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `hip_pitch_amplitude` | 0.34 rad | 0.45 rad | 增大髋关节摆动幅度，使左右脚交替超越更明显 |
| `knee_pitch_amplitude` | 0.46 rad | 0.60 rad | 增大膝关节摆动幅度，提高摆动相抬膝高度 |
| `swing_knee_scale` | 1.10 | 1.35 | 摆动相额外膝关节缩放，进一步提高抬膝 |
| `stance_ratio` | 0.60 | 0.55 | 缩短支撑相比例，延长摆动相，脚在空中时间更长 |
| `scale`（残差幅度） | 0.08 | 0.12 | 策略残差预算增大，允许更大幅度的修正 |

参考轨迹本身驱动明显的交替迈步，策略只需学习小残差修正，降低了学习难度。

#### 奖励目标（`velocity_env_cfg.py`）

| 奖励项 | 参数 | 旧值 | 新值 | 说明 |
|--------|------|------|------|------|
| `hip_alternation` | `target_separation` | 0.36 rad | 0.50 rad | 要求更大的左右髋角差，强制交替步态 |
| `knee_flexion` | `target` | 0.28 rad | 0.38 rad | 引导更高的平均抬膝角度 |
| `feet_air_time` | `threshold` | 0.3 s | 0.2 s | 降低离地时间门槛，更容易触发离地奖励 |

---

## [0.4.0] - 2026-04-18

### Fixed — 训练崩溃（std 负值）+ 步态不对称（跛行）

#### 问题 1：`RuntimeError: normal expects all elements of std >= 0.0`

**根因**：`rsl_rl_ppo_cfg.py` 使用了已废弃的 `stochastic=True, init_noise_std=1.0` 参数，经 `handle_deprecated_rsl_rl_cfg` 迁移后生成的 `GaussianDistribution` 默认使用 `std_type="scalar"`，即直接将标准差作为可学习参数。训练过程中梯度更新可能使该参数变为负值，导致 `torch.distributions.Normal` 报错崩溃。

**修复**（`rsl_rl_ppo_cfg.py`）：弃用 `stochastic` / `init_noise_std` 旧参数，直接使用 `distribution_cfg` 新 API，并指定 `std_type="log"`。`"log"` 空间下网络学习 `log(std)`，实际标准差为 `exp(log_std) > 0` 恒成立，彻底消除负值风险。

```python
# 旧写法（已废弃，std 可能变负）
actor = RslRlMLPModelCfg(stochastic=True, init_noise_std=1.0, ...)

# 新写法（std 恒正）
actor = RslRlMLPModelCfg(
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
        init_std=1.0,
        std_type="log",
    ),
    ...
)
```

#### 问题 2：左脚偏低、右脚抬得更多（跛行步态）

**根因**：`hip_alternation` 奖励权重仅 0.5，`knee_flexion` 奖励只考察平均膝角而不惩罚左右差异，策略发现不对称步态同样能前进。

**修复**：
- `hip_alternation` 权重 0.5 → 2.0，强制左右髋关节反相运动
- `knee_flexion` 权重 0.35 → 0.8
- 新增 `knee_symmetry_penalty`（weight=-1.0）：直接惩罚左右膝关节位置差的平方，消除单腿主导现象
- `feet_air_time` 权重 0.5 → 1.5，鼓励双脚均匀离地

### Added
- `rewards.py`：新增 `knee_symmetry_penalty` 函数，惩罚左右膝关节不对称

---

## [0.3.0] - 2026-04-18

### Fixed — `play.py` 兼容 `rsl_rl >= 5.0` + 绕圈步态

#### 问题 1：`TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'`

**根因**：`rsl_rl >= 5.0` 将 `stochastic` / `init_noise_std` 从 `MLPModel.__init__()` 移除，改用 `distribution_cfg`。`train.py` 已通过 `handle_deprecated_rsl_rl_cfg` 自动迁移，但 `play.py` 遗漏了这一调用。

**修复**（`play.py`）：
```python
import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
```

#### 问题 2：`AttributeError: 'PPO' object has no attribute 'actor_critic'`

**根因**：`rsl_rl >= 5.0` 将 actor/critic 拆分，`PPO` 不再有 `actor_critic` 属性，`obs_normalizer` 也内嵌进 `MLPModel`。旧的 `export_policy_as_jit(ppo_runner.alg.actor_critic, ...)` 调用失效。

**修复**（`play.py`）：改用 runner 内置导出方法：
```python
# 旧写法
export_policy_as_jit(ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, ...)

# 新写法
ppo_runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
ppo_runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
```

#### 问题 3：`ValueError: too many values to unpack` at `obs, _ = env.get_observations()`

**根因**：新版 `RslRlVecEnvWrapper.get_observations()` 返回单个 `TensorDict`，不再是 `(obs, extras)` 元组。

**修复**（`play.py`）：
```python
# 旧写法
obs, _ = env.get_observations()

# 新写法
obs = env.get_observations()
```

#### 问题 4：机器人绕圈走（偏航不受约束）

**根因**：奖励函数中没有偏航角速度惩罚。`forward_progress` 和 `track_lin_vel_x_exp` 均基于机体坐标系 x 方向速度，机器人原地旋转时 body-frame x 速度仍为正，策略因此学到绕圈步态。

**修复**：
- `rewards.py`：新增 `ang_vel_z_l2` 函数，惩罚偏航角速度平方
- `velocity_env_cfg.py`：加入 `yaw_rate` 惩罚项（weight=-0.5），侧向速度惩罚从 -0.08 加强到 -0.3

#### 速度课程调整（`flat_env_cfg.py`）

| 参数 | 旧值 | 新值 |
|------|------|------|
| 起始速度 | 0.18 m/s | 0.10 m/s |
| 最大速度 | 0.26 m/s | 0.30 m/s |

降低起始速度让机器人先在低速下学会直行，再逐步提速。

#### PPO 配置更新（`rsl_rl_ppo_cfg.py`）

- `obs_groups` 从空字典改为 `{"actor": ["policy"], "critic": ["policy"]}`，消除训练时的 UserWarning
- `max_iterations` 300 → 500，给新奖励结构更多收敛时间
- `entropy_coef` 0.0 → 0.005，增加探索性，避免过早收敛到局部最优

### Added
- `rewards.py`：新增 `ang_vel_z_l2` 偏航角速度惩罚函数

---

## [0.2.0] - 2026-04-16

### Added
- **Step 5** — Guguji-specific flat/rough env configs (`config/guguji/flat_env_cfg.py`, `rough_env_cfg.py`)
  - `GugujiRoughEnvCfg` / `GugujiRoughEnvCfg_PLAY`: wires `GUGUJI_CFG`, 2048 envs, reference gait action, gait phase obs
  - `GugujiFlatEnvCfg` / `GugujiFlatEnvCfg_PLAY`: flat plane, no height scan, velocity curriculum starting at 0.18 m/s
- **Step 5** — Guguji PPO agent configs (`config/guguji/agents/rsl_rl_ppo_cfg.py`)
  - `GugujiFlatPPORunnerCfg`: net=[256,256,128], lr=6e-5, clip=0.15, ent_coef=0.0 (matches `walk_ppo.yaml`)
  - `GugujiRoughPPORunnerCfg`: net=[512,256,128], 1500 iterations
- **Step 6** — Gym environment registration (`config/guguji/__init__.py`)
  - `Isaac-Velocity-Flat-Guguji-v0`, `Isaac-Velocity-Flat-Guguji-Play-v0`
  - `Isaac-Velocity-Rough-Guguji-v0`, `Isaac-Velocity-Rough-Guguji-Play-v0`
  - Wired into import chain: `velocity/__init__.py` → `config/__init__.py` → `config/guguji/__init__.py`
- **Step 7** — Reference gait `ActionTerm` (`mdp/actions.py`)
  - `ReferenceGaitAction` / `ReferenceGaitActionCfg`: vectorised PyTorch port of the Gazebo sinusoidal gait
  - Per-env phase tracking; left/right legs 180° out of phase; resets on episode end
  - Policy outputs residual joint positions added on top of the reference trajectory
- **Step 7** — Gait phase observation (`mdp/observations.py`)
  - `gait_phase_obs`: exposes `(sin, cos)` of current gait phase to the policy
- **Step 8** — Velocity command curriculum (`mdp/curriculums.py`)
  - `velocity_command_curriculum`: continuously advances command velocity upper bound (0.18 → 0.26 m/s) as tracking success improves, mirroring the 3-stage `walk_ppo.yaml` curriculum

### Changed
- `mdp/__init__.py`: exports `ReferenceGaitAction`, `ReferenceGaitActionCfg`, `gait_phase_obs`, `velocity_command_curriculum`
- `velocity/__init__.py`: imports `config` sub-package to trigger Gym registrations on package load
- `config/__init__.py`: imports `guguji` sub-package

---

## [0.1.0] - 2026-04-16

### Added
- Initialized repository from `IsaacLabExtensionTemplate`, renamed to `guguji_locomotion`
- `README.md` and `README_ZH.md` with project overview and migration context
- `CHANGELOG.md`
- Parent repo `.gitignore` updated to exclude `guguji_isaaclab/` (independent git sub-project)

### Migration Context
- Source project: `guguji_rl` (Stable-Baselines3 + Gazebo Fortress + ROS 2 Jazzy)
- Target: Isaac Lab 2.1.0 + RSL-RL + Isaac Sim 4.5.0
- Robot: Guguji biped, 8 joints (hip/knee/ankle_pitch + ankle per leg)
- Trained policies in source: balance (200k steps) + walk curriculum (3 stages, up to 0.26 m/s)


### Added
- **Step 5** — Guguji-specific flat/rough env configs (`config/guguji/flat_env_cfg.py`, `rough_env_cfg.py`)
  - `GugujiRoughEnvCfg` / `GugujiRoughEnvCfg_PLAY`: wires `GUGUJI_CFG`, 2048 envs, reference gait action, gait phase obs
  - `GugujiFlatEnvCfg` / `GugujiFlatEnvCfg_PLAY`: flat plane, no height scan, velocity curriculum starting at 0.18 m/s
- **Step 5** — Guguji PPO agent configs (`config/guguji/agents/rsl_rl_ppo_cfg.py`)
  - `GugujiFlatPPORunnerCfg`: net=[256,256,128], lr=6e-5, clip=0.15, ent_coef=0.0 (matches `walk_ppo.yaml`)
  - `GugujiRoughPPORunnerCfg`: net=[512,256,128], 1500 iterations
- **Step 6** — Gym environment registration (`config/guguji/__init__.py`)
  - `Isaac-Velocity-Flat-Guguji-v0`, `Isaac-Velocity-Flat-Guguji-Play-v0`
  - `Isaac-Velocity-Rough-Guguji-v0`, `Isaac-Velocity-Rough-Guguji-Play-v0`
  - Wired into import chain: `velocity/__init__.py` → `config/__init__.py` → `config/guguji/__init__.py`
- **Step 7** — Reference gait `ActionTerm` (`mdp/actions.py`)
  - `ReferenceGaitAction` / `ReferenceGaitActionCfg`: vectorised PyTorch port of the Gazebo sinusoidal gait
  - Per-env phase tracking; left/right legs 180° out of phase; resets on episode end
  - Policy outputs residual joint positions added on top of the reference trajectory
- **Step 7** — Gait phase observation (`mdp/observations.py`)
  - `gait_phase_obs`: exposes `(sin, cos)` of current gait phase to the policy
- **Step 8** — Velocity command curriculum (`mdp/curriculums.py`)
  - `velocity_command_curriculum`: continuously advances command velocity upper bound (0.18 → 0.26 m/s) as tracking success improves, mirroring the 3-stage `walk_ppo.yaml` curriculum

### Changed
- `mdp/__init__.py`: exports `ReferenceGaitAction`, `ReferenceGaitActionCfg`, `gait_phase_obs`, `velocity_command_curriculum`
- `velocity/__init__.py`: imports `config` sub-package to trigger Gym registrations on package load
- `config/__init__.py`: imports `guguji` sub-package

---

## [0.1.0] - 2026-04-16

### Added
- Initialized repository from `IsaacLabExtensionTemplate`, renamed to `guguji_locomotion`
- `README.md` and `README_ZH.md` with project overview and migration context
- `CHANGELOG.md`
- Parent repo `.gitignore` updated to exclude `guguji_isaaclab/` (independent git sub-project)

### Migration Context
- Source project: `guguji_rl` (Stable-Baselines3 + Gazebo Fortress + ROS 2 Jazzy)
- Target: Isaac Lab 2.1.0 + RSL-RL + Isaac Sim 4.5.0
- Robot: Guguji biped, 8 joints (hip/knee/ankle_pitch + ankle per leg)
- Trained policies in source: balance (200k steps) + walk curriculum (3 stages, up to 0.26 m/s)
