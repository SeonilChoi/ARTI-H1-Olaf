# Olaf walk real2sim (2026-11-21)

Replays the real walk segments of humanoid_controller `arti_h1_olaf_run_*.csv` logs in sim, with the same g sequence from the
stand -> walk switch, and compares per-segment metrics (in place / forward / turn / 2.5 s after a turn): gyro std and band
power, roll / pitch, |q_des - q| and torque rms per joint type, knee saturation, double support.

    python scripts/real2sim/walk_replay/walk_replay.py --checkpoint <walk.pt> --num_envs 32 --tag nominal --logs <csv...> --headless
    R2S='cfg.dr_start_kp=(0.8,0.9)' python .../walk_replay.py ... --tag kp   # cfg overrides
    python scripts/real2sim/walk_replay/compare.py nominal kp

Result (walk_10 agent_480000, hard-floor logs 095329 / 095641 / 100031, 4 walk segments, 3 real falls):
- old cfg: 0 / 128 sim falls. Real |q_des - q| 1.3-1.8x sim at 0.6-0.9x torque -> +15 ms actuator delay and kp x0.8-0.9.
- real gyro 2-3x sim at 3-12 Hz (roll / pitch rate) and 0-6 Hz (yaw rate); no delay / gain / Coulomb / friction / hip-yaw
  stiffness change reproduces it -> torso_torque_noise_std (2.0, 2.5, 2.0) Nm, bands (3-12, 3-12, 0.5-5 Hz) matches the std.
- every real fall came ~2 s after a left turn ended, with yaw-rate and pitch oscillation growing (gz std 0.3 -> 1.0).
