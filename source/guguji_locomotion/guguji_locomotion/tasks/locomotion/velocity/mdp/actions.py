"""Reference gait ActionTerm for Guguji biped.

Implements the same sinusoidal reference gait from guguji_rl/envs/gazebo_biped_env.py
as an Isaac Lab ActionTerm. The policy outputs residual joint positions that are
added on top of the reference trajectory.

Usage in env config:
    actions.joint_pos = mdp.ReferenceGaitActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.08,
        use_default_offset=True,
        gait_period=0.72,
        stance_ratio=0.60,
        hip_pitch_amplitude=0.34,
        hip_pitch_bias=0.04,
        knee_pitch_amplitude=0.46,
        knee_pitch_bias=0.12,
        swing_knee_scale=1.10,
        ankle_pitch_amplitude=0.22,
        ankle_pitch_bias=-0.05,
        push_off_ankle_scale=0.22,
    )
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ReferenceGaitAction(ActionTerm):
    """Joint position action with a sinusoidal reference gait bias.

    The final joint target is:
        q_target = q_default + reference_offset(phase) + scale * policy_action

    The gait phase is advanced each env step by ``step_dt / gait_period``.
    Left and right legs are 180° out of phase.
    """

    cfg: ReferenceGaitActionCfg
    _asset: Articulation

    def __init__(self, cfg: ReferenceGaitActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._scale = cfg.scale

        # Resolve joint indices once
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        # Gait parameters
        self._gait_period = max(cfg.gait_period, env.step_dt)
        self._stance_ratio = float(torch.clamp(torch.tensor(cfg.stance_ratio), 0.05, 0.95))
        self._hip_amp = cfg.hip_pitch_amplitude
        self._hip_bias = cfg.hip_pitch_bias
        self._knee_amp = cfg.knee_pitch_amplitude
        self._knee_bias = cfg.knee_pitch_bias
        self._swing_knee_scale = cfg.swing_knee_scale
        self._ankle_amp = cfg.ankle_pitch_amplitude
        self._ankle_bias = cfg.ankle_pitch_bias
        self._push_off_ankle_scale = cfg.push_off_ankle_scale

        # Map joint names to indices within the action vector
        def _find(name: str) -> int | None:
            for i, jn in enumerate(self._joint_names):
                if jn == name:
                    return i
            return None

        self._left_hip_idx = _find("left_hip_pitch_joint")
        self._right_hip_idx = _find("right_hip_pitch_joint")
        self._left_knee_idx = _find("left_knee_pitch_joint")
        self._right_knee_idx = _find("right_knee_pitch_joint")
        self._left_ankle_idx = _find("left_ankle_pitch_joint")
        self._right_ankle_idx = _find("right_ankle_pitch_joint")

        # Per-env gait phase, shape (num_envs,)
        self._gait_phase = torch.zeros(env.num_envs, device=env.device)

        # Processed actions buffer
        self._processed_actions = torch.zeros(
            env.num_envs, self._num_joints, device=env.device
        )

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Compute reference offsets and add scaled policy residuals."""
        self._raw_actions = actions.clone()

        # Default (nominal) joint positions, shape (num_envs, num_joints)
        default_pos = self._asset.data.default_joint_pos[:, self._joint_ids]

        # Reference gait offsets, shape (num_envs, num_joints)
        ref_offsets = self._compute_reference_offsets()

        # Residual from policy
        residual = actions * self._scale

        self._processed_actions = default_pos + ref_offsets + residual

    def apply_actions(self) -> None:
        """Send joint position targets to the articulation."""
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset gait phase for the given environments."""
        self._gait_phase[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_phase(self) -> None:
        """Advance gait phase by one env step."""
        dt_over_period = self._env.step_dt / self._gait_period
        self._gait_phase = (self._gait_phase + dt_over_period) % 1.0

    def _gait_profile(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised gait profile for a batch of phases.

        Returns (hip_profile, knee_profile, ankle_profile), each shape (num_envs,).
        """
        sr = self._stance_ratio
        in_stance = phase < sr

        # Stance phase
        stance_progress = phase / sr
        hip_stance = 1.0 - 2.0 * stance_progress
        knee_stance = 0.18 * torch.sin(math.pi * stance_progress)
        push_off_progress = torch.clamp((stance_progress - 0.60) / 0.40, min=0.0)
        ankle_stance = -0.70 * hip_stance + self._push_off_ankle_scale * push_off_progress

        # Swing phase
        swing_progress = (phase - sr) / (1.0 - sr)
        hip_swing = -1.0 + 2.0 * swing_progress
        knee_swing = self._swing_knee_scale * torch.sin(math.pi * swing_progress)
        ankle_swing = -0.45 * hip_swing - 0.25 * torch.sin(math.pi * swing_progress)

        hip_profile = torch.where(in_stance, hip_stance, hip_swing)
        knee_profile = torch.where(in_stance, knee_stance, knee_swing)
        ankle_profile = torch.where(in_stance, ankle_stance, ankle_swing)

        return hip_profile, knee_profile, ankle_profile

    def _compute_reference_offsets(self) -> torch.Tensor:
        """Compute per-env reference gait offsets, shape (num_envs, num_joints)."""
        self._advance_phase()

        offsets = torch.zeros(
            self._env.num_envs, self._num_joints, device=self._env.device
        )

        left_phase = self._gait_phase % 1.0
        right_phase = (self._gait_phase + 0.5) % 1.0

        left_hip_p, left_knee_p, left_ankle_p = self._gait_profile(left_phase)
        right_hip_p, right_knee_p, right_ankle_p = self._gait_profile(right_phase)

        def _set(idx: int | None, value: torch.Tensor) -> None:
            if idx is not None:
                offsets[:, idx] = value

        _set(self._left_hip_idx, self._hip_bias + self._hip_amp * left_hip_p)
        _set(self._right_hip_idx, self._hip_bias + self._hip_amp * right_hip_p)
        _set(self._left_knee_idx, self._knee_bias + self._knee_amp * left_knee_p)
        _set(self._right_knee_idx, self._knee_bias + self._knee_amp * right_knee_p)
        _set(self._left_ankle_idx, self._ankle_bias + self._ankle_amp * left_ankle_p)
        _set(self._right_ankle_idx, self._ankle_bias + self._ankle_amp * right_ankle_p)

        return offsets


@configclass
class ReferenceGaitActionCfg(ActionTermCfg):
    """Configuration for the reference gait action term."""

    class_type: type = ReferenceGaitAction

    asset_name: str = MISSING
    """Name of the articulation asset in the scene."""

    joint_names: list[str] = MISSING
    """Regex patterns or exact names of joints to control."""

    scale: float = 0.08
    """Scale applied to the policy residual output."""

    # Gait parameters (from walk_ppo.yaml)
    gait_period: float = 0.72
    """Full gait cycle duration in seconds."""

    stance_ratio: float = 0.60
    """Fraction of the cycle spent in stance phase."""

    hip_pitch_amplitude: float = 0.34
    """Peak hip pitch offset (rad)."""

    hip_pitch_bias: float = 0.04
    """Constant hip pitch offset added to the reference (rad)."""

    knee_pitch_amplitude: float = 0.46
    """Peak knee pitch offset (rad)."""

    knee_pitch_bias: float = 0.12
    """Constant knee pitch offset (rad)."""

    swing_knee_scale: float = 1.10
    """Extra scale on knee flexion during swing phase."""

    ankle_pitch_amplitude: float = 0.22
    """Peak ankle pitch offset (rad)."""

    ankle_pitch_bias: float = -0.05
    """Constant ankle pitch offset (rad)."""

    push_off_ankle_scale: float = 0.22
    """Scale for push-off ankle extension at end of stance."""
