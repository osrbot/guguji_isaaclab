"""Custom observation functions for Guguji biped locomotion.

All functions follow the Isaac Lab MDP convention:
    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  shape (num_envs, dim)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current gait phase as (sin, cos) pair, shape (num_envs, 2).

    Reads the phase from the ReferenceGaitAction term named ``joint_pos``.
    Falls back to zeros if the action term is not present or not a ReferenceGaitAction.
    """
    from .actions import ReferenceGaitAction

    action_term = env.action_manager.get_term("joint_pos")
    if isinstance(action_term, ReferenceGaitAction):
        phase = action_term._gait_phase  # (num_envs,)
        angle = 2.0 * torch.pi * phase
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    return torch.zeros(env.num_envs, 2, device=env.device)
