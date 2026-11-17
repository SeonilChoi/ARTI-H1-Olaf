# ARTI-H1 Olaf (KB robot)

The ARTI-H1 knee-bent (KB) robot trained with the method of Müller, Knoop et al., *Olaf: Bringing an Animated
Character to Life in the Physical World* (arXiv 2512.16705v2). This project is separate from `arti_h1_local`. It copies
the KB robot asset and the measured robot models from there (`tasks/direct/base/`, as of 2026-11-17) and does not
import anything from it.

## What the paper's method looks like here

| Paper | Here |
|---|---|
| Separate walking and standing policies | `Arti-H1-Olaf-Walk-v0`, `Arti-H1-Olaf-Stand-v0` (`-Play-` variants: 16 envs) |
| Path frame `p_PF = (x, y, ψ)`: integrates the path velocity when walking, converges to the feet midpoint when standing, stays within a bound of the torso | `olaf_env.py` `_pre_physics_step` (0.10 m / 0.35 rad bound, τ = 0.5 s when standing) |
| Animation reference from a gait tool with heel-toe motion, `x_t = f(p_PF, g, φ)` | PlaCo walk (KB stance) with an added heel-toe roll (`reference/gen_walk_clips.py`), 250 clips on a step grid, phase-binned relative to the clip's path frame (`reference/build_reference.py`); looked up by path velocity (4-NN blend) and phase |
| Standing input `g = (neck, torso orientation, height)` | `g = (dz, roll, pitch, yaw)` of the trunk. PlaCo IK grid 4×5×5×5 (`reference/gen_stand_grid.py`), multilinear lookup. No neck |
| `g` randomised over its full range | Walk: uniform box, projected onto the reference coverage (vx −0.12..0.15, vy ±0.068 m/s, wz ±0.4 rad/s). Stand: uniform with a rate-limited follow |
| Policy state (eq. 3) | Root pose relative to the path frame, root-frame velocities, q, q̇, a(t−1), a(t−2), actuator temperatures, φ (walk), then g |
| Rewards (Tab. I) with weights (Tab. III) | All except the neck terms. The angular-velocity-xy weight is missing from Tab. III, so the z weight (1.5) is used |
| Thermal model (eq. 6) + CBF reward (eq. 5c), `T_max` 80 °C, clip [70, 85] °C | Applied to the 12 leg actuators. α and β are fitted on the real logs (below). γ_T comes from the paper's feasibility rule (0.0108). Initial temperatures are spread up to 85 °C |
| Joint-limit CBF (q_m 0.1 rad, γ_q 20), foot–foot collision, saturated impact reduction | Real motor limits. Collision is an oriented sole-rectangle overlap test. Impact: Δv_z between physics steps in cm/s, capped at 50 cm/s |
| PPO, actor and critic 3×512, privileged critic, actor noise, pushes | `agents/skrl_olaf_ppo_cfg.yaml`. 8192 envs on the server (local 4096 / 2048), 100k iterations |
| Runtime (Sec. VII-C): policy 50 Hz, first-order hold to the 600 Hz actuator rate, 37.5 Hz low-pass | Sim at 600 Hz with decimation 12. Each physics step ramps linearly between the last two policy targets, applies the 5.24 rad/s slew limit and the 37.5 Hz low-pass (`_apply_action`). The controller runs the same chain in a 600 Hz servo thread |
| State estimate: contact-aided invariant EKF on IMU + leg kinematics [42] | `olaf/inekf.py`, batched and run in the training loop for the actor. It uses the MTi-3 gyro, a simulated accelerometer (10 ms mean specific force, noise 0.3 m/s², bias ±0.2 m/s²) and the leg kinematics. The C++ twin is `robot::OlafInEKF`, which agrees to 1e-14 |

Left out, because the robot does not have them: the battery model, the upper body (neck, arms, mouth, eyes, eyebrows,
show functions) and the Animation Engine.

## Real-to-sim (existing ARTI-H1 logs)

* Thermal: 28 robot logs (humanoid_controller git history plus working tree, 10-24..11-16, about 50 min at 60 Hz) are
  listed in `data/real_log_index.txt`. The fit is `scripts/thermal/fit_thermal.py` and the result is
  `assets/thermal/arti_h1_thermal.json`. The leg motors heat only 0–6 °C per session, so α and T_amb cannot be
  identified from these logs. α is therefore fixed at 0.001 s⁻¹ (time constant about 17 min), T_amb is drawn from 20–31 °C
  (the observed session starts), and β is fitted per joint type.
* Actuator delay, jitter and extra lag, Coulomb friction, armature, PD-gain DR, MTi-3 gyro noise, bias and delay, joint
  zero errors, yaw torque bias, pushes and the 5.24 rad/s target rate limit are the KB line's real2sim recipe
  (arti_h1_local `logs/rsi_kb`, kb05–kb13, from the 11-13..11-16 real logs). The KB line's per-step target low-pass
  (0.6 at 60 Hz) is replaced by the paper's runtime chain above. The IMU delay stays one control step (now 20 ms).

## Commands

```bash
# references (motion_generator venv has PlaCo)
/opt/isaac/projects/motion_generator/.venv/bin/python reference/gen_walk_clips.py --out reference/clips/walk --jobs 12
/opt/isaac/projects/motion_generator/.venv/bin/python reference/gen_stand_grid.py --out reference/clips/stand_grid.npz
/opt/isaac/IsaacLab-main/.venv/bin/python reference/build_reference.py --walk reference/clips/walk \
    --stand reference/clips/stand_grid.npz --out source/arti_h1_olaf/arti_h1_olaf/assets/reference/olaf_kb_reference.npz

# install (once) and train
uv pip install --python /opt/isaac/IsaacLab-main/.venv/bin/python --no-deps -e source/arti_h1_olaf
python scripts/skrl/train.py --task Arti-H1-Olaf-Walk-v0 physics=isaacsim_physx --num_envs 4096 agent.agent.experiment.directory=olaf_walk
python scripts/skrl/train.py --task Arti-H1-Olaf-Stand-v0 physics=isaacsim_physx --num_envs 4096 agent.agent.experiment.directory=olaf_stand

# reference sanity check (open-loop playback of the reference)
python scripts/check_reference.py --task Arti-H1-Olaf-Walk-Play-v0 --no_dr --headless
```

## Server (Isaac Sim PhysX, tmux)

Same layout as the local machine: Isaac Lab venv at `/opt/isaac/IsaacLab-main/.venv`, projects under `/opt/isaac/projects`.
The reference table (`assets/reference/olaf_kb_reference.npz`) and the thermal parameters are committed, so the
server needs neither PlaCo nor the real logs.

```bash
# once
cd /opt/isaac/projects
git clone https://github.com/SeonilChoi/ARTI-H1-Olaf.git arti_h1_olaf
source /opt/isaac/IsaacLab-main/.venv/bin/activate
cd arti_h1_olaf
uv pip install --no-deps -e source/arti_h1_olaf   # or: python -m pip install --no-deps -e source/arti_h1_olaf

# every terminal
source /opt/isaac/IsaacLab-main/.venv/bin/activate
cd /opt/isaac/projects/arti_h1_olaf && git pull
export OMNI_KIT_ACCEPT_EULA=yes

# walking policy (paper: 8192 envs, 100k iterations = 2.4M skrl timesteps)
tmux new -s olaf-walk
python scripts/skrl/train.py --task Arti-H1-Olaf-Walk-v0 physics=isaacsim_physx --num_envs 8192 \
    agent.agent.experiment.directory=olaf_walk
# standing policy (second terminal)
tmux new -s olaf-stand
python scripts/skrl/train.py --task Arti-H1-Olaf-Stand-v0 physics=isaacsim_physx --num_envs 8192 \
    agent.agent.experiment.directory=olaf_stand
# resume: add --checkpoint logs/skrl/olaf_walk/<run>/checkpoints/agent_N.pt
```

Or all of it in one tmux session `olaf` (windows tensorboard / walk / stand; installs the package if needed):

```bash
cd /opt/isaac/projects/arti_h1_olaf && git pull
bash scripts/server_tmux.sh               # 8192 envs each; e.g. "bash scripts/server_tmux.sh 4096 4096" on one GPU
tmux attach -t olaf                       # Ctrl+b n = next window, Ctrl+b d = detach
```

Runs land in `logs/skrl/olaf_walk|olaf_stand/<date>_ppo_torch/` (checkpoint every 1000 iterations). TensorBoard:
`tensorboard --logdir logs/skrl --port 6006`. Watch `Olaf/joint_err_deg`, `Olaf/torso_ori_err_deg`,
`Olaf/torso_pos_err_cm`, `Olaf/contact_match`, `Olaf/temp_over_max_frac`, `Episode / Total timesteps (mean)` (max 1199
= no falls in a 20 s episode) and the `Reward/<term>` scalars.

Video of a checkpoint (headless is the default, mp4 in `<run>/videos/play/`):

```bash
python scripts/skrl/play.py --task Arti-H1-Olaf-Walk-Play-v0 --checkpoint <run>/checkpoints/agent_N.pt \
    --num_envs 4 --video --video_length 600 physics=isaacsim_physx
```

## Deployment (humanoid_controller branch `arti-h1-olaf`)

`robot_olaf.yaml` (`policy.observation: olaf`) runs both policies. The robot starts standing; on the pad, cross
switches to walking and circle to standing.

The controller builds the paper's state inputs itself:

* **State estimate.** The pelvis orientation, position and velocity come from the contact-aided invariant EKF
  (`robot::OlafInEKF` in `ArtiH1::olaf_estimate`), as in the paper.
* **Runtime chain.** The policy runs at 50 Hz. A 600 Hz servo thread applies the first-order hold, the slew limit and
  the 37.5 Hz low-pass, then writes the motor commands.
* **Path frame, clock and control input g.** These follow `ArtiH1OlafEnv`.
* **Motor temperatures.** They are read from the bus.

The env builds the actor's root pose, orientation and velocity with the same filter (`actor_state_estimator`); the
critic and the rewards use the truth. `scripts/check_estimator.py` compares the filter with the truth in closed loop
(`--ideal` = noiseless IMU). With zero actions while standing it converges to under 0.005 m/s and about 0.6° of tilt
(the accelerometer bias and the tilt cannot be told apart when the robot is still).

Export after training:

```bash
python scripts/export_onnx.py --checkpoint logs/skrl/olaf_walk/<run>/checkpoints/agent_N.pt \
    --out <humanoid_controller>/config/robot/ARTI-H1/agent_olaf_walk.onnx \
    --keys_out <humanoid_controller>/config/robot/ARTI-H1/olaf_walk_keys.yaml
python scripts/export_onnx.py --checkpoint logs/skrl/olaf_stand/<run>/checkpoints/agent_N.pt \
    --out <humanoid_controller>/config/robot/ARTI-H1/agent_olaf_stand.onnx
```

The script checks that the ONNX output matches the torch policy.
