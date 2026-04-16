"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    command_name: str,
    min_vel: float,
    max_vel: float,
    success_threshold: float = 0.8,
    success_steps: int = 200,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gradually increase the commanded forward velocity range as the robot improves.

    Mirrors the 3-stage curriculum in walk_ppo.yaml (0.18 -> 0.22 -> 0.26 m/s) but
    operates continuously: the upper bound of the velocity command range is raised by
    ``step_size`` whenever the mean tracking success over the last ``success_steps``
    episodes exceeds ``success_threshold``.

    The curriculum state is stored on the env object under ``_vel_curriculum_max_vel``.

    Args:
        env: The RL environment.
        env_ids: Environment indices being reset.
        command_name: Name of the velocity command term.
        min_vel: Minimum forward velocity (lower bound, kept fixed).
        max_vel: Maximum forward velocity the curriculum can reach.
        success_threshold: Fraction of envs that must be tracking well to advance.
        success_steps: How many steps to average success over (not used directly here;
            advancement is checked each curriculum call).
        asset_cfg: Robot articulation config.

    Returns:
        Current maximum commanded velocity (scalar tensor).
    """
    # Retrieve or initialise curriculum state
    if not hasattr(env, "_vel_curriculum_max_vel"):
        env._vel_curriculum_max_vel = min_vel  # type: ignore[attr-defined]
        env._vel_curriculum_success_buf = torch.zeros(  # type: ignore[attr-defined]
            env.num_envs, device=env.device
        )

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_x = command[env_ids, 0]
    vel_x = asset.data.root_lin_vel_b[env_ids, 0]

    # Mark success: tracking within 0.05 m/s of command
    tracking_ok = (torch.abs(vel_x - cmd_x) < 0.05).float()
    env._vel_curriculum_success_buf[env_ids] = tracking_ok  # type: ignore[attr-defined]

    mean_success = env._vel_curriculum_success_buf.mean().item()  # type: ignore[attr-defined]

    step_size = 0.02  # advance 0.02 m/s at a time
    if mean_success >= success_threshold:
        env._vel_curriculum_max_vel = min(  # type: ignore[attr-defined]
            env._vel_curriculum_max_vel + step_size, max_vel  # type: ignore[attr-defined]
        )

    # Update the command range
    cmd_term = env.command_manager.get_term(command_name)
    if hasattr(cmd_term, "cfg") and hasattr(cmd_term.cfg, "ranges"):
        cmd_term.cfg.ranges.lin_vel_x = (min_vel, env._vel_curriculum_max_vel)  # type: ignore[attr-defined]

    return torch.tensor(env._vel_curriculum_max_vel, device=env.device)  # type: ignore[attr-defined]

