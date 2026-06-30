"""World-coordinate aim -> VPT-compatible discrete camera action.

The #1 silent-bug surface of the project. Three rules enforced here:
  1. pitch is DOWN-positive (Minecraft convention).
  2. the camera field is a per-tick RELATIVE delta, never an absolute angle.
  3. degrees go straight into the mu-law binner -- never * CAMERA_SCALER.
"""
import math
import numpy as np

from . import constants as C


class CameraQuantizer:
    """Faithful port of OpenAI VPT lib/actions.py camera quantizer (mu_law)."""

    def __init__(self, maxval=C.CAMERA_MAXVAL, binsize=C.CAMERA_BINSIZE,
                 mu=C.CAMERA_MU, scheme=C.CAMERA_QUANTIZATION):
        self.maxval, self.binsize, self.mu, self.scheme = maxval, binsize, mu, scheme

    def discretize(self, deg):
        deg = np.clip(np.asarray(deg, dtype=np.float64), -self.maxval, self.maxval)
        if self.scheme == "mu_law":
            x = deg / self.maxval
            x = np.sign(x) * np.log(1.0 + self.mu * np.abs(x)) / np.log(1.0 + self.mu)
            deg = x * self.maxval
        return np.round((deg + self.maxval) / self.binsize).astype(np.int64)

    def undiscretize(self, b):
        deg = np.asarray(b, dtype=np.float64) * self.binsize - self.maxval
        if self.scheme == "mu_law":
            x = deg / self.maxval
            x = np.sign(x) / self.mu * ((1.0 + self.mu) ** np.abs(x) - 1.0)
            deg = x * self.maxval
        return deg


def bin_centers_deg(q: "CameraQuantizer" = None):
    q = q or CameraQuantizer()
    return q.undiscretize(np.arange(C.CAMERA_N_BINS))


def wrap180(a):
    """Wrap degrees to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def aim_angles(shooter_eye, target_point):
    """Return (yaw_deg, pitch_deg) to look from shooter_eye at target_point.
    Minecraft convention: yaw 0=+Z increasing toward -X; pitch DOWN-positive."""
    dx = target_point[0] - shooter_eye[0]
    dy = target_point[1] - shooter_eye[1]
    dz = target_point[2] - shooter_eye[2]
    yaw = math.degrees(-math.atan2(dx, dz))
    pitch = math.degrees(-math.atan2(dy, math.hypot(dx, dz)))
    return yaw, pitch


def relative_camera_delta(cur_yaw, cur_pitch, tgt_yaw, tgt_pitch,
                          max_step=C.CAMERA_MAXVAL):
    """Per-tick (d_pitch, d_yaw) toward the target look angles, yaw-wrapped and
    clipped to +-max_step. Large turns must be issued over multiple ticks."""
    d_yaw = wrap180(tgt_yaw - cur_yaw)
    d_pitch = tgt_pitch - cur_pitch
    d_yaw = max(-max_step, min(max_step, d_yaw))
    d_pitch = max(-max_step, min(max_step, d_pitch))
    return d_pitch, d_yaw


def camera_to_bins(d_pitch, d_yaw, q: "CameraQuantizer" = None):
    """(d_pitch, d_yaw) degrees -> (pitch_bin, yaw_bin, combined_index).
    combined = pitch_bin * N_BINS + yaw_bin  (VPT convention)."""
    q = q or CameraQuantizer()
    pb = int(q.discretize(d_pitch))
    yb = int(q.discretize(d_yaw))
    return pb, yb, pb * C.CAMERA_N_BINS + yb


def world_to_camera_action(shooter_eye, target_point, cur_yaw, cur_pitch,
                           q: "CameraQuantizer" = None):
    """Full one-tick pipeline. Returns the target angles, the clipped per-tick
    delta, the discrete bins, and the residual still to cover (so the caller
    knows whether to keep turning next tick)."""
    q = q or CameraQuantizer()
    tgt_yaw, tgt_pitch = aim_angles(shooter_eye, target_point)
    d_pitch, d_yaw = relative_camera_delta(cur_yaw, cur_pitch, tgt_yaw, tgt_pitch)
    pb, yb, idx = camera_to_bins(d_pitch, d_yaw, q)
    resid_yaw = wrap180(tgt_yaw - cur_yaw) - d_yaw
    resid_pitch = (tgt_pitch - cur_pitch) - d_pitch
    return {
        "target_yaw": tgt_yaw, "target_pitch": tgt_pitch,
        "camera_deg": (d_pitch, d_yaw),        # order: [pitch, yaw]
        "camera_bins": (pb, yb), "camera_index": idx,
        "residual_deg": (resid_pitch, resid_yaw),
        "aligned": abs(resid_yaw) < 0.29 and abs(resid_pitch) < 0.29,
    }


# --- Bow charge ---
def bow_power(hold_ticks):
    f = hold_ticks / 20.0
    f = (f * f + 2.0 * f) / 3.0
    return min(f, 1.0)


def arrow_speed(hold_ticks):
    return bow_power(hold_ticks) * C.BOW_FULL_CHARGE_SPEED


def is_fireable(hold_ticks):
    return bow_power(hold_ticks) >= 0.1
