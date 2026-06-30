"""Tick-accurate Minecraft arrow simulator + ballistic pitch solver.

Reference oracle for the M0 physics-parity gate. Per-tick update order
(minecraft.wiki/Arrow): velocity *= drag; velocity.y -= gravity; position += velocity.
VERIFY this order against the live game in M0 before trusting it for labels.
"""
import math
import numpy as np

from . import constants as C


def look_vector(yaw_deg, pitch_deg):
    """Minecraft look unit vector for (yaw, pitch) in degrees (pitch down-positive)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    return np.array([
        -math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    ])


def simulate_arrow(speed, pitch_deg, yaw_deg=0.0, start=(0.0, 0.0, 0.0),
                   max_ticks=400, drag=C.ARROW_DRAG, gravity=C.ARROW_GRAVITY):
    """Return an (max_ticks+1, 3) array of arrow positions, starting at `start`."""
    pos = np.array(start, dtype=np.float64)
    vel = look_vector(yaw_deg, pitch_deg) * speed
    traj = [pos.copy()]
    for _ in range(max_ticks):
        vel = vel * drag
        vel[1] -= gravity
        pos = pos + vel
        traj.append(pos.copy())
    return np.array(traj)


def height_at_distance(speed, pitch_deg, horiz_dist, **kw):
    """Interpolated y when the arrow first reaches horiz_dist in the xz-plane.
    Returns None if it never gets that far."""
    traj = simulate_arrow(speed, pitch_deg, **kw)
    prev = traj[0]
    for cur in traj[1:]:
        hz_prev = math.hypot(prev[0], prev[2])
        hz_cur = math.hypot(cur[0], cur[2])
        if hz_cur >= horiz_dist:
            if hz_cur == hz_prev:
                return float(cur[1])
            t = (horiz_dist - hz_prev) / (hz_cur - hz_prev)
            return float(prev[1] + t * (cur[1] - prev[1]))
        prev = cur
    return None


def solve_pitch(speed, horiz_dist, height_delta, lo=-45.0, hi=30.0, iters=60):
    """Pitch (deg, down-positive) so a freshly-loosed arrow passes through
    (horiz_dist, height_delta) -- the lower (direct) arc. Bisection on a
    function that is monotonically decreasing in pitch."""
    def f(p):
        h = height_at_distance(speed, p, horiz_dist)
        return None if h is None else h - height_delta

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if fm is None or fm > 0:   # aimed too far up (or fell short) -> bigger pitch
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def lead_target(speed, shooter_eye, mob_pos, mob_vel, aim_offset=0.0, iters=5):
    """Fixed-point lead: predict where the arrow intercepts a constant-velocity
    mob. Returns the predicted aim point (x, y, z). aim_offset raises the point
    to the mob hitbox centre. mob_vel is blocks/tick."""
    target = np.array(mob_pos, dtype=np.float64) + np.array([0.0, aim_offset, 0.0])
    eye = np.array(shooter_eye, dtype=np.float64)
    vel = np.array(mob_vel, dtype=np.float64)
    for _ in range(iters):
        horiz = math.hypot(target[0] - eye[0], target[2] - eye[2])
        # crude time-of-flight estimate using the drag-decayed mean horizontal speed
        tof = horiz / max(speed * 0.85, 1e-6)
        target = np.array(mob_pos, dtype=np.float64) + vel * tof + np.array([0.0, aim_offset, 0.0])
    return tuple(target)
