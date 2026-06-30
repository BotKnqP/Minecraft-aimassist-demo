"""High-frequency bearing tracker — decouples the control rate from the detection rate.

The detector runs at ~10 fps (CPU YOLO ceiling, even with GPU it's bound by the mod's
capture interval). The mod ticks at 20 Hz. If Python only emits an action when a fresh
detection arrives, the control rate is stuck at the detector's rate and you get the
'two/three-shot acquire' feel: detect -> turn -> re-detect (overshot) -> correct ->
re-detect (still off) -> correct again.

This tracker keeps the LATEST KNOWN bearing of the committed target in CURRENT view
coordinates. Two things keep it accurate:
  * on a fresh detection it SNAPS to the measured bearing
  * after each commanded turn it SUBTRACTS the turn the mod is about to apply (mirrored
    gain + deadzone + clamp) from the held bearing, so the next emit at 20 Hz reflects
    the view that mod will be at one tick from now

For static aim_botz targets (no target motion), this is enough — the held bearing is
exact between detections. For moving targets we'd add an alpha-beta velocity term later;
parking that for v0.4.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


# Mirror BowMacroController constants. Keep in sync — these are read-only here, the
# authoritative copies live in the mod (see mod/.../oracle/BowMacroController.java).
MOD_TURN_GAIN = 0.45
MOD_LIVE_MAX_STEP_DEG = 10.0
MOD_MOVE_DEADZONE_DEG = 1.2


def expected_mod_turn(commanded_deg: float) -> float:
    """How many degrees the mod will ACTUALLY turn given a commanded relative delta. Mirrors
    BowMacroController.stepView (deadzone -> 0; otherwise gain * delta, clipped to ±max step)."""
    if abs(commanded_deg) < MOD_MOVE_DEADZONE_DEG:
        return 0.0
    moved = commanded_deg * MOD_TURN_GAIN
    if moved > MOD_LIVE_MAX_STEP_DEG:
        return MOD_LIVE_MAX_STEP_DEG
    if moved < -MOD_LIVE_MAX_STEP_DEG:
        return -MOD_LIVE_MAX_STEP_DEG
    return moved


@dataclass
class _State:
    bearing_dy: float        # CURRENT-VIEW bearing of the target (yaw, deg, +right)
    bearing_dp: float        # CURRENT-VIEW bearing (pitch, deg, +down)
    bbox_h: float            # for the k/h range model
    bbox_w: float            # carried so we can rebuild a Detection for the ESP overlay
    conf: float              # last seen confidence
    last_meas_ts_ms: float   # when we last got a fresh measurement


class TargetState:
    """Holds the committed target's bearing in CURRENT view. Updated by fresh detections,
    decayed by commanded turns. is_stale(now) reports if the held value is too old to trust."""

    def __init__(self, max_predict_ms: int = 300, send_history_ms: int = 1000):
        self.max_predict_ms = int(max_predict_ms)
        self.send_history_ms = int(send_history_ms)
        self._st = None     # _State or None
        # ring of (send_unix_ms, expected_mod_yaw_turn_deg, expected_mod_pitch_turn_deg).
        # Used to back-correct a fresh detection: subtract every turn that happened AFTER the
        # frame's capture_ms so the bearing reflects the CURRENT view, not the stale capture view.
        self._sends: deque = deque()

    # ---------------- public API ---------------------------------------------------------

    def has_target(self, now_ms: float = None) -> bool:
        if self._st is None:
            return False
        if now_ms is None:
            now_ms = time.monotonic() * 1000.0
        return (now_ms - self._st.last_meas_ts_ms) <= self.max_predict_ms

    def on_measurement(self, d_yaw: float, d_pitch: float, bbox_h: float,
                       bbox_w: float, conf: float, now_ms: float,
                       capture_ms: float = None) -> None:
        """A fresh detection arrived — snap the held bearing to it. The measurement was made on
        a frame CAPTURED at `capture_ms` (mod wall-clock ms); since then the mod has been
        turning per our commands. Subtract the cumulative expected mod-turn AFTER capture so the
        snapped bearing reflects the CURRENT view, not the stale capture view. capture_ms=None
        falls back to legacy "trust the raw measurement" behaviour (used by old 'R' frames + tests)."""
        adj_dy = float(d_yaw)
        adj_dp = float(d_pitch)
        if capture_ms is not None:
            # subtract every expected mod-turn from sends that happened AT OR AFTER capture_ms
            for ts, dy_t, dp_t in self._sends:
                if ts >= capture_ms:
                    adj_dy -= dy_t
                    adj_dp -= dp_t
        self._st = _State(
            bearing_dy=adj_dy,
            bearing_dp=adj_dp,
            bbox_h=float(bbox_h),
            bbox_w=float(bbox_w),
            conf=float(conf),
            last_meas_ts_ms=float(now_ms),
        )

    def on_send(self, sent_d_yaw: float, sent_d_pitch: float, now_ms: float = None) -> None:
        """Account for the turn the mod is about to apply: subtract the EXPECTED actual
        movement (gain + clamp + deadzone) from our held bearing so the NEXT emit reflects
        the view we'll be at one tick from now. Also append (ts, expected_y, expected_p) to the
        send history so a later on_measurement can back-correct a frame that was captured
        before some of these sends. No-op when no target is held."""
        if now_ms is None:
            now_ms = time.time() * 1000.0  # wall clock (matches mod's System.currentTimeMillis)
        ey = expected_mod_turn(sent_d_yaw)
        ep = expected_mod_turn(sent_d_pitch)
        # prune old entries
        cutoff = now_ms - self.send_history_ms
        while self._sends and self._sends[0][0] < cutoff:
            self._sends.popleft()
        self._sends.append((float(now_ms), ey, ep))
        if self._st is None:
            return
        self._st.bearing_dy -= ey
        self._st.bearing_dp -= ep

    def current_bearing(self) -> tuple:
        """(d_yaw, d_pitch, bbox_h, bbox_w, conf). Caller should check has_target() first."""
        if self._st is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        return (self._st.bearing_dy, self._st.bearing_dp,
                self._st.bbox_h, self._st.bbox_w, self._st.conf)

    def age_ms(self, now_ms: float) -> float:
        return float("inf") if self._st is None else (now_ms - self._st.last_meas_ts_ms)

    def reset(self) -> None:
        self._st = None
