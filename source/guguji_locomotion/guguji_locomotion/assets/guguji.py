"""Guguji biped robot ArticulationCfg for Isaac Lab.

Robot specs (from URDF):
  - 8 revolute joints: hip_pitch / knee_pitch / ankle_pitch / ankle  (left + right)
  - Total mass: ~3.4 kg
  - Joint effort limits: 10 Nm (hip/knee), 8 Nm (ankle)
  - Joint velocity limits: 2.0 rad/s

URDF path is resolved relative to the guguji_simulation workspace root.
If the layout changes, update GUGUJI_URDF_PATH below.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# guguji_isaaclab/source/guguji_locomotion/guguji_locomotion/assets/guguji.py
#   -> up 5 levels -> guguji_simulation/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = os.path.normpath(os.path.join(_THIS_DIR, *[".."] * 5))
GUGUJI_URDF_PATH = os.path.join(
    _WS_ROOT,
    "guguji_ros2_ws", "src", "guguji_ros2", "urdf", "guguji.urdf",
)

# ---------------------------------------------------------------------------
# Nominal joint positions (rad) — from walk_ppo.yaml
# ---------------------------------------------------------------------------
GUGUJI_NOMINAL_JOINT_POS = {
    "left_hip_pitch_joint":   0.04,
    "left_knee_pitch_joint":  0.18,
    "left_ankle_pitch_joint": -0.10,
    "left_ankle_joint":       0.00,
    "right_hip_pitch_joint":  0.04,
    "right_knee_pitch_joint": 0.18,
    "right_ankle_pitch_joint": -0.10,
    "right_ankle_joint":      0.00,
}

# ---------------------------------------------------------------------------
# ArticulationCfg
# ---------------------------------------------------------------------------
GUGUJI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=GUGUJI_URDF_PATH,
        # USD cache is written next to the URDF on first run
        usd_dir=os.path.join(os.path.dirname(GUGUJI_URDF_PATH), "usd_cache"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),  # spawn height — base ~0.32 m above ground at nominal pose
        joint_pos=GUGUJI_NOMINAL_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Hip and knee: higher stiffness, more torque
        "hip_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_pitch_joint", ".*knee_pitch_joint"],
            effort_limit=10.0,
            velocity_limit=2.0,
            stiffness=80.0,   # kp — needs tuning in sim
            damping=4.0,      # kd
        ),
        # Ankle pitch: medium stiffness
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_pitch_joint"],
            effort_limit=8.0,
            velocity_limit=2.0,
            stiffness=40.0,
            damping=2.0,
        ),
        # Ankle roll: softer
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_joint"],
            effort_limit=8.0,
            velocity_limit=2.0,
            stiffness=30.0,
            damping=1.5,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
