"""Base locomotion velocity-tracking environment for Guguji biped.

Adapted from the IsaacLabExtensionTemplate for a biped robot:
  - Forward-only velocity commands (no lateral / yaw for initial training)
  - Roll / pitch / height termination conditions
  - Biped-specific reward terms (upright, height, hip alternation, knee flexion)
  - Foot contact sensor uses *foot_link body names
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import guguji_locomotion.tasks.locomotion.velocity.mdp as mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Terrain scene with Guguji biped."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    # Height scanner attached to base_link
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # Track all bodies; feet are filtered by body name in reward/termination terms
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Forward-only velocity commands for initial biped training."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.3),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Joint position residual actions around the nominal pose."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.08,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization events — expanded for sim2sim and sim2real robustness."""

    # ---- startup: randomized once per training run, simulates hardware variability ----

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),   # wider range for sim2real
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )
    # Randomize mass of all links to simulate manufacturing tolerances
    add_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    # Actuator gain randomization — most critical for sim2real (motor variability ±20%)
    randomize_hip_knee_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_pitch_joint", ".*knee_pitch_joint"]),
            "stiffness_distribution_params": (64.0, 96.0),   # kp: 80 ± 20%
            "damping_distribution_params": (3.2, 4.8),        # kd: 4 ± 20%
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_ankle_pitch_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch_joint"]),
            "stiffness_distribution_params": (32.0, 48.0),   # kp: 40 ± 20%
            "damping_distribution_params": (1.6, 2.4),        # kd: 2 ± 20%
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_ankle_roll_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_joint"]),
            "stiffness_distribution_params": (24.0, 36.0),   # kp: 30 ± 20%
            "damping_distribution_params": (1.2, 1.8),        # kd: 1.5 ± 20%
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    # Joint friction and armature — simulates real joint stiction and rotor inertia
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.0, 0.05),
            "armature_distribution_params": (0.0, 0.01),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # COM offset randomization — simulates payload placement and assembly tolerances
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # ---- reset: randomized at each episode reset ----

    # External force/torque applied at reset — simulates payload and assembly offsets
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-3.0, 3.0),
            "torque_range": (-1.0, 1.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.85, 1.15),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ---- interval: periodic disturbances during the episode ----

    # Stronger and more frequent pushes
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.3, 0.3)}},
    )
    # Gravity perturbation — simulates IMU bias and slight slope effects
    randomize_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.05, 0.05, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for Guguji biped locomotion."""

    # -- task
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=4.8,
        params={"command_name": "base_velocity", "std": 0.10},
    )
    forward_progress = RewTerm(
        func=mdp.forward_progress,
        weight=6.0,
        params={},
    )
    alive_bonus = RewTerm(func=mdp.is_alive, weight=0.6)

    # -- upright / posture
    upright = RewTerm(func=mdp.upright_reward, weight=1.6)
    height = RewTerm(
        func=mdp.height_reward,
        weight=0.9,
        params={"target_height": 0.32},
    )

    # -- gait quality
    hip_alternation = RewTerm(
        func=mdp.hip_alternation_reward,
        weight=2.0,
        params={
            "left_hip_name": "left_hip_pitch_joint",
            "right_hip_name": "right_hip_pitch_joint",
            "target_separation": 0.50,   # larger stride target (was 0.36)
            "antiphase_sigma": 0.20,
        },
    )
    knee_flexion = RewTerm(
        func=mdp.knee_flexion_reward,
        weight=0.8,
        params={
            "left_knee_name": "left_knee_pitch_joint",
            "right_knee_name": "right_knee_pitch_joint",
            "target": 0.38,   # higher knee lift target (was 0.28)
            "sigma": 0.15,
        },
    )
    knee_symmetry = RewTerm(
        func=mdp.knee_symmetry_penalty,
        weight=-2.0,
        params={
            "left_knee_name": "left_knee_pitch_joint",
            "right_knee_name": "right_knee_pitch_joint",
        },
    )
    hip_symmetry = RewTerm(
        func=mdp.hip_symmetry_penalty,
        weight=-1.5,
        params={
            "left_hip_name": "left_hip_pitch_joint",
            "right_hip_name": "right_hip_pitch_joint",
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link"),
            "command_name": "base_velocity",
            "threshold": 0.2,   # easier to trigger (was 0.3)
        },
    )

    # -- penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.004)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.05)
    lateral_velocity = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.3)
    yaw_rate = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.5)
    backward_velocity = RewTerm(func=mdp.backward_velocity_penalty, weight=-2.8)
    stall_penalty = RewTerm(
        func=mdp.stall_penalty,
        weight=-4.6,
        params={"command_name": "base_velocity", "threshold": 0.10},
    )
    undesired_knee_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*knee.*"),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions for Guguji biped."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Base body hits the ground
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )
    # Robot tilts too far (roll or pitch > ~0.9 rad)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.9},
    )
    # Base drops below minimum height
    base_height = DoneTerm(
        func=mdp.base_height_below_threshold,
        params={"minimum_height": 0.21},
    )


@configclass
class CurriculumCfg:
    """Terrain difficulty curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Base locomotion env for Guguji on rough terrain."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4                  # policy at 50 Hz (physics at 200 Hz)
        self.episode_length_s = 20.0
        self.sim.dt = 0.005                  # 200 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
