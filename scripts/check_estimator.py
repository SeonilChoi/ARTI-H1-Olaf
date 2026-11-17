"""Closed-loop check of the robot-side observation (humanoid_controller arti-h1-olaf, ArtiH1::olaf_estimate /
olaf_advance) in sim: the policy runs on observations built exactly like the controller builds them (IMU attitude +
stance-foot kinematic odometry, path frame / phase / g integrated in that estimator frame), instead of the env's
ground-truth observations.

    python scripts/check_estimator.py --checkpoint <run>/checkpoints/agent_N.pt --task Arti-H1-Olaf-Walk-Play-v0 \
        [--num_envs 32] [--steps 1200] [--truth]

Reports falls, pelvis velocity vs the commanded path velocity, and the RMS difference between the controller-style and
the env observation blocks (position, rotation, velocity). --truth runs the policy on the env observation (baseline).
"""

import argparse
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="Arti-H1-Olaf-Walk-Play-v0")
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--truth", action="store_true")
ap.add_argument("--trace", type=int, default=-1)
ap.add_argument("--contact_tol", type=float, default=0.012)
ap.add_argument("--vel_tau", type=float, default=0.03)
AppLauncher.add_app_launcher_args(ap)
a, _ = ap.parse_known_args()
app = AppLauncher(a)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import arti_h1_olaf.tasks  # noqa: F401, E402

cfg = resolve_task_config(a.task, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = a.num_envs
cfg.episode_length_s = 1e4
cfg.command_resample_time_s = (4.0, 8.0)
env = gym.make(a.task, cfg=cfg)
u = env.unwrapped
dev, n, dt = u.device, u.num_envs, u.step_dt

c = torch.load(a.checkpoint, map_location=dev, weights_only=False)
P, pre = c["policy"], c["observation_preprocessor"]
idx = sorted({int(k.split(".")[1]) for k in P if k.startswith("net_container.") and k.endswith(".weight")})


def policy(o):
    x = ((o - pre["running_mean"].float()) / (torch.sqrt(pre["running_variance"].float()) + 1e-8)).clamp(-5, 5)
    for j, i in enumerate(idx):
        x = x @ P[f"net_container.{i}.weight"].T + P[f"net_container.{i}.bias"]
        if j < len(idx) - 1:
            x = torch.nn.functional.elu(x)
    return x


def yaw_of_R(R):
    return torch.atan2(R[:, 1, 0], R[:, 0, 0])


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class Mirror:
    """Python twin of ArtiH1::olaf_estimate / olaf_advance / the olaf observation."""

    def __init__(self):
        self.init = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pos = torch.zeros(n, 3, device=dev)
        self.vel = torch.zeros(n, 3, device=dev)
        self.anchor = torch.zeros(n, 2, 3, device=dev)
        self.contact = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.path = torch.zeros(n, 3, device=dev)
        self.phase = torch.full((n,), cfg.walk_start_phase, device=dev)

    def estimate(self, R, rel, feet_yaw):
        self.R, self.rel, self.feet_yaw, self.root_yaw = R, rel, feet_yaw, yaw_of_R(R)
        new = ~self.init
        if new.any():
            self.pos[new] = 0.0
            self.pos[new, 2] = -0.5 * (rel[new, 0, 2] + rel[new, 1, 2])
            self.anchor[new] = self.pos[new, None] + rel[new]
            self.anchor[new, :, 2] = 0.0
            self.contact[new] = True
            self.vel[new] = 0.0
            self.path[new] = torch.stack((self.pos[new, 0], self.pos[new, 1], self.root_yaw[new]), -1)
            self.init[:] = True
        zmin = rel[..., 2].min(-1, keepdim=True).values
        contact = rel[..., 2] - zmin < a.contact_tol
        td = contact & ~self.contact
        anc = self.pos[:, None] + rel
        anc[..., 2] = 0.0
        self.anchor = torch.where(td[..., None], anc, self.anchor)
        self.contact = contact
        p = ((self.anchor - rel) * contact[..., None]).sum(1) / contact.sum(-1, keepdim=True).clamp_min(1)
        alpha = dt / (dt + a.vel_tau)
        self.vel += alpha * ((p - self.pos) / dt - self.vel)
        self.pos = p
        self.feet = self.pos[:, None] + rel

    def obs(self, gyro_b, qrel, qd, a1, a2, temp_obs, g, walk):
        psi = self.path[:, 2]
        cs, sn = torch.cos(psi), torch.sin(psi)
        dx, dy = self.pos[:, 0] - self.path[:, 0], self.pos[:, 1] - self.path[:, 1]
        p_rel = torch.stack((cs * dx + sn * dy, -sn * dx + cs * dy, self.pos[:, 2]), -1)
        Rz = torch.zeros(n, 3, 3, device=dev)
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cs, sn, -sn, cs, 1.0
        rot6 = (Rz @ self.R)[..., :2].reshape(n, 6)
        lin_b = (self.R.transpose(1, 2) @ self.vel[..., None])[..., 0]
        tail = [torch.stack((torch.cos(2 * math.pi * self.phase), torch.sin(2 * math.pi * self.phase)), -1), g] if walk else [g]
        return torch.cat((p_rel, rot6, lin_b, gyro_b, qrel, qd, a1, a2, temp_obs, *tail), -1)

    def advance(self, g, walk):
        if walk:
            self.phase = (self.phase + dt / cfg.walk_period_override) % 1.0
            psi = self.path[:, 2]
            self.path[:, 0] += (torch.cos(psi) * g[:, 0] - torch.sin(psi) * g[:, 1]) * dt
            self.path[:, 1] += (torch.sin(psi) * g[:, 0] + torch.cos(psi) * g[:, 1]) * dt
            self.path[:, 2] = psi + g[:, 2] * dt
        else:
            yl, yr = self.feet_yaw[:, 0], self.feet_yaw[:, 1] + math.pi
            psi_f = torch.atan2(torch.sin(yl) + torch.sin(yr), torch.cos(yl) + torch.cos(yr))
            k = 1.0 - math.exp(-dt / cfg.stand_path_tau_s)
            mid = self.feet[:, :, :2].mean(1)
            self.path[:, :2] += k * (mid - self.path[:, :2])
            self.path[:, 2] += k * wrap(psi_f - self.path[:, 2])
        e = self.path[:, :2] - self.pos[:, :2]
        dist = e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.path[:, :2] = self.pos[:, :2] + e * torch.clamp(cfg.path_max_distance / dist, max=1.0)
        self.path[:, 2] = self.root_yaw + torch.clamp(wrap(self.path[:, 2] - self.root_yaw), -cfg.path_max_yaw, cfg.path_max_yaw)


cfg.walk_period_override = None
with torch.inference_mode():
    obs, _ = env.reset()
    cfg.walk_period_override = u.walk_period
    m = Mirror()
    if u.walk:
        m.phase = u.phase.clone()
    d = u.robot.data
    pidx = u.sim_joint_ids_from_policy if hasattr(u, "sim_joint_ids_from_policy") else torch.arange(12, device=dev)
    falls, diff, vel_err, cnt = 0, torch.zeros(3, device=dev), 0.0, 0
    never_reset = torch.ones(n, dtype=torch.bool, device=dev)
    for t in range(a.steps):
        quat_imu, ang_w_imu = u.mti3_imu.read(d, u.imu_body_index)
        R = matrix_from_quat(quat_imu)
        pel_p, pel_q = d.body_pos_w.torch[:, u.ref_body_index], d.body_quat_w.torch[:, u.ref_body_index]
        feet = u._foot_positions()
        Rp = matrix_from_quat(pel_q)
        sole_b = (Rp.transpose(1, 2)[:, None] @ (feet - pel_p[:, None])[..., None])[..., 0]  # FK: soles in the pelvis frame
        rel = (R[:, None] @ sole_b[..., None])[..., 0]
        fyaw = torch.stack([yaw_of_R(R @ Rp.transpose(1, 2) @ matrix_from_quat(d.body_quat_w.torch[:, bi]))
                            for bi in u.foot_body_indices], -1)
        if t == 2:  # the env reset's IMU state settles after the first steps (startup artefact of this check only)
            m.init[:] = False
            if u.walk:
                m.phase = u.phase.clone()
        m.estimate(R, rel, fyaw)
        env_obs = obs["policy"]
        mo = m.obs(env_obs[:, 12:15], env_obs[:, 15:27], env_obs[:, 27:39], env_obs[:, 39:51], env_obs[:, 51:63],
                   env_obs[:, 63:75], u.g, u.walk)
        if t >= 3:  # only envs that never reset (the twin re-anchors differently after a reset)
            k = never_reset.float()[:, None]
            nk = never_reset.sum().clamp_min(1)
            diff += torch.stack((((mo[:, 0:3] - env_obs[:, 0:3]).square() * k).sum() / (3 * nk),
                                 ((mo[:, 3:9] - env_obs[:, 3:9]).square() * k).sum() / (6 * nk),
                                 ((mo[:, 9:12] - env_obs[:, 9:12]).square() * k).sum() / (3 * nk)))
        if a.trace >= 0 and t % 6 == 0 and t < 120:
            e = a.trace
            f = lambda x: " ".join(f"{v:+.3f}" for v in x.tolist())  # noqa: E731
            print(f"t{t:3d} p env [{f(env_obs[e, 0:3])}] mir [{f(mo[e, 0:3])}] | rot env [{f(env_obs[e, 3:9])}] mir [{f(mo[e, 3:9])}]"
                  f" | v env [{f(env_obs[e, 9:12])}] mir [{f(mo[e, 9:12])}] c {m.contact[e].tolist()} ph {u.phase[e]:.2f}/{m.phase[e]:.2f}")
        if a.trace >= 0 and t < 8:
            e = a.trace
            print(f"  dbg t{t} m.path {[round(x, 3) for x in m.path[e].tolist()]} m.pos {[round(x, 3) for x in m.pos[e].tolist()]} "
                  f"m.yaw {m.root_yaw[e]:.3f} | u.path {[round(x, 3) for x in u.path[e].tolist()]} true yaw "
                  f"{yaw_of_R(matrix_from_quat(pel_q))[e]:.3f} pel {[round(x, 3) for x in pel_p[e].tolist()]} g {[round(x, 3) for x in u.g[e].tolist()]}")
        act = policy(env_obs if a.truth else mo)
        obs, _, term, trunc, _ = env.step(act)
        m.advance(u.g, u.walk)
        never_reset &= ~(term | trunc)
        if term.any():
            falls += int(term.sum())
            m.init[term] = False
            m.phase[term] = u.phase[term] if u.walk else 0.0
        if u.walk:
            v = d.body_lin_vel_w.torch[:, u.ref_body_index]
            cp, sp = torch.cos(u.path[:, 2]), torch.sin(u.path[:, 2])
            lin = torch.stack((cp * v[:, 0] + sp * v[:, 1], -sp * v[:, 0] + cp * v[:, 1]), -1)
            vel_err += (lin - u.g[:, :2]).norm(dim=-1).mean().item()
        cnt += 1
    rms = torch.sqrt(diff / max(cnt - 3, 1)).tolist()
    print(f"[check_estimator] envs never reset: {int(never_reset.sum())}/{n}")
    msg = (f"[check_estimator] {'TRUTH obs' if a.truth else 'controller-style obs'}: falls {falls} in {n} envs x {a.steps} steps; "
           f"obs RMS diff vs env: pos {rms[0]:.4f} m, rot {rms[1]:.4f}, vel {rms[2]:.4f} m/s")
    if u.walk:
        msg += f"; |v_xy - g| (path frame) {vel_err / cnt:.3f} m/s"
    print(msg)
env.close()
app.app.close()
