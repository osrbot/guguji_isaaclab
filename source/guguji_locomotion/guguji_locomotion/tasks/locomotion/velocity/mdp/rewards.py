"""Custom reward functions for Guguji biped locomotion.

All functions follow the Isaac Lab MDP convention:
    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  shape (num_envs,)

Ported from guguji_rl/guguji_rl/rewards.py.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Existing template reward (kept for reference / flat terrain use)
# ---------------------------------------------------------------------------

def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps using L2-kernel (quadruped-style)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward alternating single-stance steps for bipeds.

    Encourages the robot to keep exactly one foot on the ground at a time
    and rewards time spent in that mode up to `threshold` seconds.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


# ---------------------------------------------------------------------------
# Velocity tracking
# ---------------------------------------------------------------------------

def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking the commanded forward (x) velocity.

    reward = exp(-(v_x - cmd_x)^2 / (2 * std^2))
    """
    asset = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    vel_x = asset.data.root_lin_vel_b[:, 0]
    error = vel_x - cmd_x
    return torch.exp(-(error ** 2) / (2.0 * std ** 2))


def forward_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive forward displacement per step (clipped to >= 0)."""
    asset = env.scene[asset_cfg.name]
    vel_x = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(vel_x, min=0.0)


def backward_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for moving backward (positive value, weight should be negative)."""
    asset = env.scene[asset_cfg.name]
    vel_x = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(-vel_x, min=0.0)


def lin_vel_y_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on lateral (y) velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty when forward velocity is below threshold while a command is active.

    Encourages the robot to actually move when commanded to do so.
    penalty = max(threshold - v_x, 0)  (only when cmd_x > 0)
    """
    asset = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    vel_x = asset.data.root_lin_vel_b[:, 0]
    positive_vel = torch.clamp(vel_x, min=0.0)
    penalty = torch.clamp(threshold - positive_vel, min=0.0)
    # Only apply when there is a forward command
    return penalty * (cmd_x > 0.01).float()


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------

def upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping the base upright.

    reward = exp(-4 * (roll^2 + pitch^2))
    Uses projected gravity to infer tilt without explicit Euler angles.
    """
    asset = env.scene[asset_cfg.name]
    # projected_gravity_b: gravity vector in base frame, shape (num_envs, 3)
    # When upright, projected_gravity_b ≈ (0, 0, -9.81)
    # Tilt is captured by the x and y components
    grav_b = asset.data.projected_gravity_b  # (num_envs, 3)
    tilt_sq = grav_b[:, 0] ** 2 + grav_b[:, 1] ** 2
    return torch.exp(-4.0 * tilt_sq)


def height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining the target base height.

    reward = exp(-8 * (h - h_target)^2)
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    error = height - target_height
    return torch.exp(-8.0 * error ** 2)


# ---------------------------------------------------------------------------
# Gait quality
# ---------------------------------------------------------------------------

def hip_alternation_reward(
    env: ManagerBasedRLEnv,
    left_hip_name: str,
    right_hip_name: str,
    target_separation: float,
    antiphase_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for left/right hip joints moving in anti-phase.

    Two components multiplied together:
      1. separation_reward  = min(|left - right| / target_separation, 1.0)
      2. antiphase_reward   = exp(-(left + right)^2 / (2 * sigma^2))

    The product is high only when the hips are far apart AND symmetric about zero.
    """
    asset = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names

    def _idx(name: str) -> int:
        return joint_names.index(name)

    left_idx = _idx(left_hip_name)
    right_idx = _idx(right_hip_name)

    left_pos = asset.data.joint_pos[:, left_idx]
    right_pos = asset.data.joint_pos[:, right_idx]

    separation = torch.abs(left_pos - right_pos)
    separation_reward = torch.clamp(separation / target_separation, max=1.0)

    sum_pos = left_pos + right_pos
    antiphase_reward = torch.exp(-(sum_pos ** 2) / (2.0 * antiphase_sigma ** 2))

    return separation_reward * antiphase_reward


def knee_flexion_reward(
    env: ManagerBasedRLEnv,
    left_knee_name: str,
    right_knee_name: str,
    target: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining average knee flexion near the target angle.

    reward = exp(-(mean_knee_abs - target)^2 / (2 * sigma^2))
    """
    asset = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names

    left_idx = joint_names.index(left_knee_name)
    right_idx = joint_names.index(right_knee_name)

    left_knee = torch.abs(asset.data.joint_pos[:, left_idx])
    right_knee = torch.abs(asset.data.joint_pos[:, right_idx])
    avg_flexion = 0.5 * (left_knee + right_knee)

    return torch.exp(-((avg_flexion - target) ** 2) / (2.0 * sigma ** 2))


def knee_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_knee_name: str,
    right_knee_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asymmetric knee flexion between left and right legs."""
    asset = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    left_idx = joint_names.index(left_knee_name)
    right_idx = joint_names.index(right_knee_name)
    diff = asset.data.joint_pos[:, left_idx] - asset.data.joint_pos[:, right_idx]
    return torch.square(diff)


def hip_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_hip_name: str,
    right_hip_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asymmetric hip pitch between left and right legs.

    Penalizes (left + right)^2, which is zero only when hips are equal and opposite.
    """
    asset = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    left_idx = joint_names.index(left_hip_name)
    right_idx = joint_names.index(right_hip_name)
    sum_pos = asset.data.joint_pos[:, left_idx] + asset.data.joint_pos[:, right_idx]
    return torch.square(sum_pos)


# ---------------------------------------------------------------------------
# Yaw / rotation penalties
# ---------------------------------------------------------------------------

def ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on yaw angular velocity to prevent spinning / circling."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


# ---------------------------------------------------------------------------
# Termination helpers (used as termination terms, not rewards)
# ---------------------------------------------------------------------------

def base_height_below_threshold(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base height drops below minimum_height (meters)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height
