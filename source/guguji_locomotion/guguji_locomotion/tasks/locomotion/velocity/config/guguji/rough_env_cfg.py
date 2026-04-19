"""Guguji biped rough-terrain locomotion environment config."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from guguji_locomotion.assets.guguji import GUGUJI_CFG
from guguji_locomotion.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import guguji_locomotion.tasks.locomotion.velocity.mdp as mdp


@configclass
class GugujiRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = GUGUJI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Guguji is a small biped — reduce env count to fit GPU memory
        self.scene.num_envs = 2048
        self.scene.env_spacing = 2.5
        # Replace plain joint position action with reference gait + residual
        self.actions.joint_pos = mdp.ReferenceGaitActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.12,           # larger residual budget for the policy
            gait_period=0.72,
            stance_ratio=0.55,    # slightly shorter stance → more air time
            hip_pitch_amplitude=0.45,   # bigger stride (was 0.34)
            hip_pitch_bias=0.04,
            knee_pitch_amplitude=0.60,  # higher knee lift (was 0.46)
            knee_pitch_bias=0.10,
            swing_knee_scale=1.35,      # more knee flexion in swing (was 1.10)
            ankle_pitch_amplitude=0.22,
            ankle_pitch_bias=-0.05,
            push_off_ankle_scale=0.25,
        )
        # Add gait phase (sin, cos) to policy observations so the policy can
        # condition its residuals on the current phase of the reference gait.
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase_obs)


@configclass
class GugujiRoughEnvCfg_PLAY(GugujiRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

