"""ARTI-H1 knee-bent (KB) robot trained the Olaf way: Mueller, Knoop et al., "Olaf: Bringing an Animated Character to
Life in the Physical World", arXiv 2512.16705v2 (Disney Research), Sec. V + VII.

What is implemented as in the paper
  * separate walking and standing policies, each its own RL problem (two tasks below)
  * path frame p_PF = (x, y, psi) (eq. 1): walking integrates the commanded path velocity, standing converges to the feet
    midpoint; it is kept within a bounded distance of the torso
  * kinematic reference x_t = f(p_PF, g, phi) (eq. 2) from a gait generator with heel-toe motion (PlaCo +
    reference/gen_walk_clips.py) for walking, from IK over g = (torso height, orientation) for standing
  * control input g randomised over its full range during training
  * actions = PD position targets; proprioceptive state s_t (eq. 3): root pose relative to the path frame, root-frame
    velocities, q, qd, a_{t-1}, a_{t-2}, actuator temperatures T, gait phase phi (walking)
  * reward = imitation + regularization + limits + impact reduction (eq. 4, Tab. I), weights Tab. III
  * thermal model (eq. 6) with a CBF reward (eq. 5c), joint-limit CBF rewards (eq. 7-8, q_m 0.1 rad, gamma_q 20),
    foot-foot collision penalty, saturated sound-suppression reward
  * early termination on torso / upper body ground contact
  * PPO, actor and critic 3 x 512, critic with privileged noiseless state, actor obs noise, random pushes

What differs (robot, not method)
  * ARTI-H1 has no neck / arms / show functions / battery module: the neck terms and the neck part of g are dropped,
    and the thermal model / reward covers the 12 leg actuators instead of the neck actuators
  * thermal coefficients and every sim-to-real property come from the existing ARTI-H1 real logs: alpha / beta fitted
    on 28 robot logs (assets/thermal/arti_h1_thermal.json), actuator delay / friction / armature / gains / IMU / joint
    zero errors / yaw bias = the KB line's real2sim recipe (arti_h1_local logs/rsi_kb, kb05-kb13)
  * control at 60 Hz (the deployed ARTI-H1 controller), not 50 Hz; 4096 envs on the 16 GB GPU instead of 8192
"""

from __future__ import annotations

import json
import os

from isaaclab.utils import configclass

from arti_h1_olaf.robots import ISAACLAB_ASSETS_DIR

from ..base.arti_h1_velocity_env_cfg import ArtiH1VelocityKBEnvCfg

REFERENCE_FILE = os.path.join(ISAACLAB_ASSETS_DIR, "reference", "olaf_kb_reference.npz")
THERMAL_FILE = os.path.join(ISAACLAB_ASSETS_DIR, "thermal", "arti_h1_thermal.json")
_THERMAL = json.load(open(THERMAL_FILE))

# PD target = default + scale * clamp(a, -1, 1): covers the reference (hip_pitch +0.5, knee -0.6, ankle_pitch +-0.45 rad
# from the KB stance) with room for balance corrections; clamped to the real joint limits afterwards.
OLAF_ACTION_SCALE_BY_TYPE = {"hip_yaw": 0.3, "hip_roll": 0.3, "hip_pitch": 0.7, "knee": 0.8, "ankle_pitch": 0.6, "ankle_roll": 0.35}
OLAF_ACTION_SCALE = {f"{s}_{j}": v for s in ("left", "right") for j, v in OLAF_ACTION_SCALE_BY_TYPE.items()}

NJ = 12
# s_t (eq. 3): p^P (3) + theta^P (6, first two columns of the rotation matrix) + v^R (3) + w^R (3) + q (12) + qd (12)
# + a_{t-1}, a_{t-2} (24) + T (12), then phi (2, walking only) and g (walk 3 / stand 4) last
_PROPRIO = 3 + 6 + 3 + 3 + NJ + NJ + 2 * NJ + NJ
WALK_OBS = _PROPRIO + 2 + 3
STAND_OBS = _PROPRIO + 4
# critic: noiseless actor terms + true temperatures / temperature rate (24) + foot heights / contacts (4) + reference
# joint error (12) and contacts (2) + torso reference error (pos 3, rot 3) + yaw torque bias + friction scale (2)
_PRIV = 2 * NJ + 4 + NJ + 2 + 6 + 2


@configclass
class ArtiH1OlafEnvCfg(ArtiH1VelocityKBEnvCfg):
    """Shared Olaf settings. Use ArtiH1OlafWalkEnvCfg / ArtiH1OlafStandEnvCfg."""

    mode = "walk"  # "walk" | "stand"
    episode_length_s = 20.0
    observation_space = WALK_OBS
    state_space = WALK_OBS + _PRIV
    reference_file = REFERENCE_FILE
    reference_knn = 4

    # --- actions ---
    action_negative_scale = OLAF_ACTION_SCALE
    action_positive_scale = OLAF_ACTION_SCALE
    obs_action_history = False  # the Olaf obs layout is built in OlafEnv

    # --- path frame (Sec. V, Fig. 4) ---
    path_max_distance = 0.10  # m, bound between the path frame and the torso (xy)
    path_max_yaw = 0.35  # rad, bound between the path frame heading and the torso heading
    stand_path_tau_s = 0.5  # standing: time constant of the convergence toward the feet midpoint

    # --- control input g, randomised across its full range (Sec. V-A) ---
    # walking: path velocity (vx, vy, wz), sampled uniformly in the box below and then projected onto the reference
    # coverage (the KNN blend of the clip keys), so the commanded path and the reference agree
    walk_command_range = ((-0.12, 0.15), (-0.068, 0.068), (-0.40, 0.40))
    walk_zero_command_fraction = 0.1
    # standing: g = (torso height dz, roll, pitch, yaw) relative to the KB stance; target resampled, then followed with a
    # rate limit (the reference stays continuous)
    stand_command_range = ((-0.03, 0.015), (-0.10, 0.10), (-0.15, 0.15), (-0.30, 0.30))
    stand_nominal_fraction = 0.2
    stand_command_rate = (0.05, 0.3, 0.3, 0.5)  # max change per second (m/s, rad/s)
    command_resample_time_s = (2.0, 6.0)

    # --- reward weights (Tab. III, Standing / Walking); neck terms dropped (no neck) ---
    w_torso_pos_xy = 4.0
    w_torso_ori = 1.5
    w_lin_vel_xy = 2.5
    w_lin_vel_z = 1.0
    # Tab. III lists no angular-vel-xy weight (Tab. I has the term): the angular-vel-z weight is used for it
    w_ang_vel_xy = 1.5
    w_ang_vel_z = 1.5
    w_leg_joint_pos = 15.0
    w_leg_joint_vel = 1.0e-3
    w_contact = 1.0
    w_survival = 20.0
    w_leg_action_rate = 5.0
    w_leg_action_acc = 1.0
    w_temperature = 2.0  # "neck temperature" weight, applied to the leg actuators
    w_joint_limits = 0.2
    w_foot_collision = 10.0
    w_impact = 2.5e-3
    w_joint_torques = 1.0e-3
    w_joint_acc = 2.5e-6
    # reward = sum(w * term) * step_dt (Isaac Lab convention; a uniform scale)
    # Tab. I kernels
    k_torso_pos = 200.0
    k_torso_ori = 20.0
    k_lin_vel = 8.0
    k_ang_vel = 2.0

    # --- joint limits CBF (Sec. V-E) ---
    joint_limit_margin = 0.1  # q_m (rad)
    joint_limit_gamma = 20.0  # gamma_q

    # --- thermal model (Sec. V-D, eq. 6) + CBF reward (eq. 5c), fitted on the real ARTI-H1 logs ---
    thermal_alpha = _THERMAL["alpha"]
    thermal_beta = _THERMAL["beta"]
    thermal_ambient_range = tuple(_THERMAL["T_ambient_range"])
    thermal_max = 80.0  # T_max (C)
    thermal_clip = (70.0, 85.0)  # [T_clip,min, T_clip,max] for observations and rewards (Sec. VII-A)
    # gamma_T: the constraint must stay feasible at the upper clip with tau = 0 (Sec. VII-A):
    # alpha (T_clip,max - T_amb,max) - gamma_T (T_clip,max - T_max) >= 0 -> computed in the env when <= 0
    thermal_gamma = 0.0
    # initial temperatures: uniform in [T_amb, thermal_init_max] per joint, so episodes start near / above T_max
    thermal_init_max = 85.0
    # sensor model: the real bus reports integer degrees
    thermal_obs_quantum = 1.0

    # --- impact reduction (Sec. V-C): -sum_feet min(dv_z^2, dv_max^2), dv_z = change of the foot vertical velocity
    # between physics steps, summed over the control step's substeps. The paper gives no unit / dv_max; in m/s the
    # term is ~1e-4 of the survival reward at Tab. III's weight, so it is evaluated in cm/s (a 0.3 m/s touchdown costs
    # ~2.3 per step at weight 2.5e-3). dv_max 50 cm/s caps contact-resolution spikes (the paper's saturation).
    impact_velocity_scale = 100.0
    impact_dv_max = 50.0

    # --- foot-foot collision: sole rectangles (foot_sole_half_length / width) overlap in xy with |dz| < this ---
    foot_collision_margin = 0.005
    foot_collision_max_dz = 0.04

    # --- actor observation noise (Sec. VII-B "noise to the actor observations"); critic is noiseless ---
    noise_root_pos = 0.01
    noise_root_rot = 0.02
    noise_lin_vel = 0.05
    noise_joint_pos = 0.01
    noise_joint_vel = 0.3
    # root pose / velocity w.r.t. the path frame come from a state estimator on the robot (paper [42]); the sim gives
    # the truth plus the noise above. The gyro is the MTi-3 model (bias / delay / noise).

    # --- reference state initialisation (walking: random phase of the commanded clip) ---
    reset_reference_state = True
    # policy switches (Sec. VII-C): walking episodes that start from the standing stance at walk_start_phase (the
    # double support before the right foot lifts, phase 0 = left single support start), standing episodes that start
    # from a mid-stride walking reference state. The robot controller starts the walking clock at walk_start_phase too.
    reset_stand_pose_fraction = 0.2
    reset_walk_pose_fraction = 0.3
    walk_start_phase = 0.92

    # --- terminations (Sec. V-C): torso / upper body on the ground -> inherited trunk-height + tilt check ---

    # --- real2sim (KB line, arti_h1_local logs/rsi_kb kb05-kb13, real logs 11-13..11-16) ---
    target_rate_limit = 5.24
    target_lowpass_alpha = 0.6
    mti3_gyro_noise_std_rad_s = 0.04
    mti3_gyro_bias_range_rad_s = (-0.03, 0.03)
    actuator_delay_step_jitter = 1
    imu_gyro_delay_per_step = True
    imu_gyro_extra_delay_steps = 1
    joint_obs_offset_range_by_type = {"hip_yaw": 0.03, "hip_roll": 0.06, "hip_pitch": 0.08, "knee": 0.08,
                                      "ankle_pitch": 0.20, "ankle_roll": 0.06}
    actuator_delay_ms = {"hip_yaw": 4.0, "hip_roll": 4.0, "hip_pitch": 4.0, "knee": 4.0, "ankle_pitch": 3.0, "ankle_roll": 3.0}
    actuator_delay_jitter_ms = (-6.0, 6.0)
    actuator_delay_extra_ms = {"hip_roll": 25.0, "ankle_pitch": 12.0}
    actuator_delay_extra_curriculum_steps = (0, 0)
    actuator_coulomb_scale_range = (0.3, 1.2)
    actuator_armature_scale_range = (0.5, 1.3)
    yaw_torque_bias_range = (-0.8, 0.8)

    def __post_init__(self):
        super().__post_init__()
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.85, 1.3)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.5, 1.2)
        self.events.push_robot.params["velocity_range"]["x"] = (-0.5, 0.5)
        self.events.push_robot.params["velocity_range"]["y"] = (-0.3, 0.3)


@configclass
class ArtiH1OlafWalkEnvCfg(ArtiH1OlafEnvCfg):
    mode = "walk"
    observation_space = WALK_OBS
    state_space = WALK_OBS + _PRIV
    # Tab. III walking column
    w_torso_pos_xy = 4.0
    w_torso_ori = 1.5
    w_lin_vel_xy = 2.5
    w_contact = 1.0
    w_leg_action_rate = 5.0
    w_leg_action_acc = 1.0
    w_joint_limits = 0.2
    w_leg_joint_vel = 1.0e-3


@configclass
class ArtiH1OlafStandEnvCfg(ArtiH1OlafEnvCfg):
    mode = "stand"
    observation_space = STAND_OBS
    state_space = STAND_OBS + _PRIV
    reset_reference_state = True
    # Tab. III standing column
    w_torso_pos_xy = 1.0
    w_torso_ori = 2.0
    w_lin_vel_xy = 1.5
    w_contact = 2.0
    w_leg_action_rate = 2.0
    w_leg_action_acc = 0.5
    w_joint_limits = 0.5
    w_leg_joint_vel = 1.0e-3


@configclass
class ArtiH1OlafWalkEnvCfg_PLAY(ArtiH1OlafWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16


@configclass
class ArtiH1OlafStandEnvCfg_PLAY(ArtiH1OlafStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16

