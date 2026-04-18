"""Guguji biped flat-terrain locomotion environment config.

Flat terrain is used for the initial curriculum stages (0.18 -> 0.22 m/s)
before transitioning to rough terrain.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import guguji_locomotion.tasks.locomotion.velocity.mdp as mdp

from .rough_env_cfg import GugujiRoughEnvCfg


@configclass
class GugujiFlatEnvCfg(GugujiRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Switch to flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # No height scan needed on flat terrain
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # Replace terrain curriculum with velocity curriculum (0.10 -> 0.30 m/s, 3 stages)
        self.curriculum.terrain_levels = None
        self.curriculum.velocity_command = CurrTerm(
            func=mdp.velocity_command_curriculum,
            params={
                "command_name": "base_velocity",
                "min_vel": 0.10,
                "max_vel": 0.30,
                "success_threshold": 0.8,
            },
        )
        # Start commands at the lowest curriculum velocity
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.10)
        # Slightly more envs since flat is cheaper
        self.scene.num_envs = 4096


@configclass
class GugujiFlatEnvCfg_PLAY(GugujiFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

