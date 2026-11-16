"""Walk reference clips for the Olaf-style imitation policy (arXiv 2512.16705, Sec. V-A).

The paper designs stylised walk cycles with a gait generation tool ([38], heel-toe motion) and maps the control input
g = path velocity to a kinematic target f(p_PF, g, phi). We do the same with PlaCo (the motion_generator project's
MotionEngine, KB stance: walk_com_height 0.29725 -> hip 0.336 / knee -0.671 / ankle -0.336 rad, pelvis z -0.0264),
plus a heel-toe roll that the plain PlaCo walk does not have:
  * late double support: the next swing foot peels off about its toe (heel-off, toes down to -toe_off),
  * swing: the foot pitch goes -toe_off -> +heel_strike while the pivot moves toe -> heel (continuous at both ends),
  * early double support: the landed foot rolls down about its heel (heel strike -> flat).
Angles scale with the forward stride (dx / walk_max_dx_forward); backward / sideways / turning steps stay flat.

Run with the motion_generator venv (PlaCo):
    /opt/isaac/projects/motion_generator/.venv/bin/python reference/gen_walk_clips.py --out reference/clips/walk --jobs 12
One <index>.npz per command grid point (dx, dy, dth per step) + grid.json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MG = os.environ.get("MOTION_GENERATOR", "/opt/isaac/projects/motion_generator")
sys.path.insert(0, os.path.join(MG, "src"))

FPS = 60
TICK = 0.001


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    return T


def _trans(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


def make_engine(gait: dict):
    from motion_generator.motion_engine import MotionEngine

    class HeelToeEngine(MotionEngine):
        """MotionEngine with a heel-toe roll on the generic_frame foot targets (walk frame, sole origin)."""

        def _foot_pitch(self, t: float) -> dict:
            out = {"left": (0.0, 0.0), "right": (0.0, 0.0)}  # side -> (pitch, toes up +, pivot x in the walk frame)
            k = float(np.clip(self._d_x / max(self._ht_dx_full, 1e-6), 0.0, 1.0))
            if k <= 0.0:
                return out
            hs, to, half, W = k * self._ht_heel, k * self._ht_toe, self._ht_half, self._ht_window
            # A pure function of the support-aligned gait phase. Not the trajectory parts (a replan restarts the current
            # part at the replan time) and not lift-off / touchdown state (MotionEngine.tick evaluates the trajectory up
            # to one planner dt ahead and back again within a tick): both made the swing progress jump.
            if t < self.periodic_start_time - W:
                return out
            P, ss = self.period, self._robot_parameters.single_support_duration
            phi = self.get_gait_phase(t)  # 0 = left single support start (right foot lifts), 0.5 = right support start
            for side, start in (("right", 0.0), ("left", 0.5)):
                tau = ((phi - start) % 1.0) * P  # time since this foot's lift-off
                if tau < ss:
                    s = smoothstep(tau / ss)
                    out[side] = (-to + (to + hs) * s, half * (1.0 - 2.0 * s))  # toe-off -> heel strike, pivot toe -> heel
                elif tau - ss < W:
                    out[side] = (hs * (1.0 - smoothstep((tau - ss) / W)), -half)  # heel strike -> flat about the heel
                elif P - tau < W:
                    out[side] = (-to * smoothstep(1.0 - (P - tau) / W), half)  # heel-off about the toe
            return out

        def _update_generic_tasks_from_trajectory(self, t: float) -> None:
            super()._update_generic_tasks_from_trajectory(t)
            p = self._foot_pitch(t)
            for side in ("left", "right"):
                pitch, px = p[side]
                if pitch == 0.0:
                    continue
                T_walk = np.asarray(getattr(self._trajectory, f"get_T_world_{side}")(t))
                T_walk = T_walk @ _trans(px) @ _rot_y(-pitch) @ _trans(-px)
                T_model = T_walk @ getattr(self, f"_T_model_from_walk_{side}")
                self._generic_tasks[f"{side}_foot"].T_world_frame = T_model

    # class attributes: MotionEngine.__init__ already calls _update_generic_tasks_from_trajectory
    HeelToeEngine._ht_heel = np.deg2rad(gait.get("heel_strike_deg", 0.0))
    HeelToeEngine._ht_toe = np.deg2rad(gait.get("toe_off_deg", 0.0))
    HeelToeEngine._ht_half = float(gait.get("heel_toe_half_length", 0.057))
    HeelToeEngine._ht_dx_full = float(gait["walk_max_dx_forward"])
    HeelToeEngine._ht_window = float(gait.get("heel_toe_window_s", 0.2))
    with contextlib.redirect_stdout(io.StringIO()):
        me = HeelToeEngine(gait["robot_folder"], gait)
    return me


def run_clip(job):
    idx, dx, dy, dth, gait, out_dir = job
    from scipy.spatial.transform import Rotation as R

    try:
        me = make_engine(gait)
        names = list(me.get_angles().keys())
        me.set_trajectory(dx, dy, dth)
        n_frames = int(round(gait["duration"] * FPS))
        q, root_p, root_q, contact, phase, valid, lf, rf = [], [], [], [], [], [], [], []
        next_rec = 1.0 / FPS
        with contextlib.redirect_stdout(io.StringIO()):
            while len(q) < n_frames:
                me.tick(TICK)
                if me.t + 1e-9 < next_rec:
                    continue
                next_rec += 1.0 / FPS
                T = np.asarray(me.robot.get_T_world_fbase())
                q.append(list(me.get_angles().values()))
                root_p.append(T[:3, 3].copy())
                root_q.append(R.from_matrix(T[:3, :3]).as_quat())  # x y z w
                contact.append(me.get_current_support_phase())
                phase.append(me.get_gait_phase())
                valid.append(float(me.is_periodic_phase_valid()))
                lf.append(np.asarray(me.get_T_world_left_contact())[:3, 3].copy())
                rf.append(np.asarray(me.get_T_world_right_contact())[:3, 3].copy())
        np.savez(os.path.join(out_dir, f"{idx}.npz"), joint_names=np.array(names), fps=FPS, q=np.array(q),
                 root_pos=np.array(root_p), root_quat=np.array(root_q), contact=np.array(contact, dtype=np.float32),
                 phase=np.array(phase), periodic_valid=np.array(valid), left_foot=np.array(lf), right_foot=np.array(rf),
                 step=np.array([dx, dy, dth]), period=me.period)
        return idx, None
    except Exception as e:  # noqa: BLE001  (PlaCo QP failures for infeasible steps)
        return idx, repr(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gait", default=os.path.join(HERE, "gait_olaf_kb.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--only", type=int, nargs="*", default=None, help="indices to (re)generate")
    a = p.parse_args()
    gait = json.load(open(a.gait))
    gait["robot_folder"] = os.path.join(HERE, "robot", gait["robot"])
    g = gait["grid"]
    grid = lambda r: np.round(np.arange(r[0], r[1] + r[2] / 2, r[2]), 5)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    table, jobs = {}, []
    for i, (dx, dy, dth) in enumerate((x, y, t) for x in grid(g["dx"]) for y in grid(g["dy"]) for t in grid(g["dth"])):
        table[i] = [float(dx), float(dy), float(dth)]
        if a.only is None or i in a.only:
            jobs.append((i, float(dx), float(dy), float(dth), gait, a.out))
    json.dump({"gait": gait, "clips": table}, open(os.path.join(a.out, "grid.json"), "w"), indent=1)
    print(f"{len(jobs)} clips -> {a.out}", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for idx, err in ex.map(run_clip, jobs):
            if err:
                failed.append(idx)
                print(f"clip {idx} {table[idx]} FAILED: {err}", flush=True)
    print(f"done, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
