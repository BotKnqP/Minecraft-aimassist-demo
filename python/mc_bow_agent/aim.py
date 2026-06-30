"""Vision -> aim logic for the scripted v1 bow agent (no RL).

Pipeline (pure functions, unit-tested):
  YOLO detections (screen bboxes) -> pick NEAREST (largest bbox) -> per-tick turn
  (d_yaw, d_pitch) to centre the crosshair on it -> add gravity-drop compensation
  from ballistic.py -> (d_yaw, d_pitch, range, fireable).

The live frame capture + socket-to-mod that drives BowMacroController is the
integration layer (separate, needs the running game). This module is the brain.

Conventions (Minecraft): +d_yaw = turn right, +d_pitch = look DOWN. Verify the
screen-X / screen-Y signs once in-game (same caveat as ProjectionUtil), then lock.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

from .ballistic import solve_pitch
from .constants import BOW_FULL_CHARGE_SPEED

ZOMBIE_HEIGHT = 1.95  # blocks (1.16.5 hitbox)

# --- FPS-aimbot selection / lock tuning ---
# Real aimbots (CS2_External GetBestTarget, GuidedHacking find_closest_target) lock the target NEAREST
# THE CROSSHAIR inside an FOV cone — NOT the largest/nearest-in-world box. Picking the biggest box made
# our view whip to ±51° screen-edge targets (even swinging to the back). These bound that.
SEL_FOV_DEG = 70.0        # default vertical FOV for selection bearings (matches runtime DEFAULT_FOV)
ACQUIRE_FOV_DEG = 40.0    # only LOCK a target within this cone of the crosshair (covers a ±34° arena
                          # spread, rejects the ±51° screen-edge box that caused the whip)
RETAIN_FOV_DEG = 50.0     # keep an already-locked target out to here (sticky > acquire; avoids edge ping-pong)
SWITCH_MARGIN_DEG = 8.0   # a challenger must beat the locked target's angular cost by this (deg) to steal it
SWITCH_COOLDOWN = 4       # min frames between switches (~0.5s at 5-10 fps); forced drops bypass it
APPROACH_STEP_DEG = 12.0  # safety ceiling on the per-frame turn toward an OFF-cone target ("look to the other
                          # side"); the mod's own 8deg/tick + 0.45 gain is the REAL rate limiter, so this is
                          # just a sanity cap (a lower value e.g. 6 double-limits and makes the pan crawl).
                          # Firing off-cone is prevented by fireable=False, NOT by this cap.


@dataclass
class Detection:
    cx: float
    cy: float
    w: float
    h: float
    conf: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @classmethod
    def from_xyxy(cls, x0, y0, x1, y1, conf):
        return cls((x0 + x1) / 2.0, (y0 + y1) / 2.0, abs(x1 - x0), abs(y1 - y0), conf)


@dataclass
class AimSolution:
    d_yaw: float          # turn right(+)/left(-), degrees
    d_pitch: float        # final pitch turn incl. drop compensation (down +, up -)
    range_blocks: float
    drop_deg: float       # degrees aimed up to compensate gravity
    fireable: bool


def pick_nearest(dets: List[Detection], conf_thresh: float = 0.5) -> Optional[Detection]:
    """Nearest zombie = largest bbox above conf_thresh. None if nothing qualifies.
    For 'shoot nearest' this also sidesteps the detector's low-conf false positives."""
    cand = [d for d in dets if d.conf >= conf_thresh]
    return max(cand, key=lambda d: d.area) if cand else None


def focal_px(frame_h: float, fov_deg: float) -> float:
    """Vertical focal length in px from the vertical FOV."""
    return (frame_h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def bearing_from_bbox(cx, cy, frame_w, frame_h, fov_deg):
    """Per-tick (d_yaw, d_pitch) degrees to centre the crosshair on (cx, cy).
    Screen centre = current look direction."""
    f = focal_px(frame_h, fov_deg)
    d_yaw = math.degrees(math.atan2(cx - frame_w / 2.0, f))
    d_pitch = math.degrees(math.atan2(cy - frame_h / 2.0, f))
    return d_yaw, d_pitch


def range_from_bbox_height(bbox_h, frame_h=None, fov_deg=None, mob_height=ZOMBIE_HEIGHT, k=None):
    """Distance (blocks) from apparent bbox height.

    Single-class makes this clean: a mob of fixed real height H spans
    px = focal * H / distance, so distance = focal * H / px = k / px.
    Prefer a calibrated k (fit from recordings via calibrate.py) over the FOV
    derivation, since the projection FOV is only approximate."""
    if bbox_h <= 0:
        return float("inf")
    if k is not None:
        return k / bbox_h
    return focal_px(frame_h, fov_deg) * mob_height / bbox_h


def aim_from_bearing(d_yaw: float, d_pitch: float, bbox_h: float, k=None,
                     max_range: float = 40.0, frame_h: float = None, fov_deg: float = None) -> AimSolution:
    """Build an AimSolution directly from a precomputed bearing (deg) + bbox height (px). Used by
    the high-freq sender in run_client, where the target's bearing is held + decayed by TargetState
    between detection frames so we don't have a fresh Detection every send."""
    rng = range_from_bbox_height(bbox_h, frame_h, fov_deg, k=k) if bbox_h > 0 else float("inf")
    drop = abs(solve_pitch(BOW_FULL_CHARGE_SPEED, max(rng if math.isfinite(rng) else max_range, 0.5), 0.0))
    return AimSolution(d_yaw=d_yaw, d_pitch=d_pitch - drop, range_blocks=rng,
                       drop_deg=drop, fireable=math.isfinite(rng) and rng <= max_range)


def aim_at(det: Detection, frame_w, frame_h, fov_deg, k=None, max_range=40.0) -> AimSolution:
    """Aim solution for one detection: centre on it, then raise for arrow drop.

    v1 approximation: assumes a roughly flat arena (target near eye level), so the
    drop compensation is |solve_pitch(speed, R, 0)| applied as extra up-pitch after
    centring. The mod, which knows the live pitch, can refine the slope later.
    Lead is omitted (zombies close radially; tangential speed is small) — TODO."""
    d_yaw, d_pitch = bearing_from_bbox(det.cx, det.cy, frame_w, frame_h, fov_deg)
    rng = range_from_bbox_height(det.h, frame_h, fov_deg, k=k)
    drop = abs(solve_pitch(BOW_FULL_CHARGE_SPEED, max(rng, 0.5), 0.0))
    return AimSolution(d_yaw=d_yaw, d_pitch=d_pitch - drop, range_blocks=rng,
                       drop_deg=drop, fireable=rng <= max_range)


def _prep_bearings(dets, frame_w, frame_h, fov_deg):
    """Precompute (det, d_yaw, d_pitch, ang_off, sel_cost) ONCE per frame — the hot paths
    (TargetTracker.select, _acquire_or_approach, the in-cone scans) used to recompute each detection's
    bearings 3-5 times with separate atan2 calls. Returns a list of tuples in dets order."""
    f = focal_px(frame_h, fov_deg)
    cx0 = frame_w / 2.0
    cy0 = frame_h / 2.0
    area_div = max(frame_w * frame_h, 1.0)
    out = []
    for d in dets:
        dy = math.degrees(math.atan2(d.cx - cx0, f))
        dp = math.degrees(math.atan2(d.cy - cy0, f))
        ang = math.hypot(dy, dp)
        cost = math.sqrt(dy * dy + 0.5 * dp * dp) - 1.5 * min((d.area / area_div) * 8.0, 1.0)
        out.append((d, dy, dp, ang, cost))
    return out


def _argmin_in_prep(prep, max_off_deg, exclude=None) -> Optional[Detection]:
    """Lowest-cost detection in a precomputed prep list whose ang_off is within max_off_deg (or None)."""
    best = None
    best_cost = float("inf")
    for d, _, _, ang, cost in prep:
        if d is exclude or ang > max_off_deg:
            continue
        if cost < best_cost:
            best_cost = cost
            best = d
    return best


def _argmin_in_cone(dets, frame_w, frame_h, fov_deg, max_off_deg, exclude=None) -> Optional[Detection]:
    """Lowest-cost detection whose centre is within `max_off_deg` of the crosshair (or None).
    Stateless entry kept for solve_from_detections / external callers; the live loop uses
    _argmin_in_prep against a cached prep list."""
    return _argmin_in_prep(_prep_bearings(dets, frame_w, frame_h, fov_deg), max_off_deg, exclude)


def pick_target(dets, frame_w, frame_h, fov_deg, conf_thresh=0.25,
                acquire_fov=ACQUIRE_FOV_DEG) -> Optional[Detection]:
    """Aimbot target acquisition: among confident detections INSIDE the FOV cone, the one nearest the
    crosshair (NOT the largest box — that whips the view to screen-edge targets). None if the cone is empty."""
    cand = [d for d in dets if d.conf >= conf_thresh]
    return _argmin_in_cone(cand, frame_w, frame_h, fov_deg, acquire_fov)


def solve_from_detections(dets, frame_w, frame_h, fov_deg, k=None,
                          conf_thresh=0.5, max_range=40.0) -> Optional[AimSolution]:
    """Stateless entry: aim at the crosshair-nearest in-cone zombie this frame (single-frame tools).
    The live loop uses TargetTracker for cross-frame commitment + switch-hysteresis."""
    target = pick_target(dets, frame_w, frame_h, fov_deg, conf_thresh)
    if target is None:
        return None
    return aim_at(target, frame_w, frame_h, fov_deg, k=k, max_range=max_range)


class TargetTracker:
    """FPS-aimbot target lock for the live loop. Picks the target NEAREST THE CROSSHAIR inside an FOV
    cone (NOT the largest / nearest-in-world box — that whipped the view to ±51° screen-edge targets and
    even swung it to the back), then commits HARD with switch-hysteresis so a flaky ~0.39-mAP detector
    can't re-flick to a different target on every dropped frame.

    Per frame (dets already conf-filtered):
      * No lock -> acquire the crosshair-nearest target in the fire cone (engaging=True). If none is in the
        cone but zombies are visible OFF to the side, lock the nearest one as an APPROACH target
        (engaging=False): the bot pans toward it ("looks to the other side") instead of idling, and fires
        once it enters the cone.
      * Lock matched (same box by the identity gate) -> stay on it; its bearing is recomputed from the new
        box. Drop only if it drifts past RETAIN_FOV. A challenger steals the lock ONLY if it is in-cone AND
        beats the locked target's cost by >= switch_margin AND the switch cooldown elapsed.
      * Lock NOT matched -> it's missing: command nothing (return None -> the mod's GRACE branch finishes
        any in-progress shot) and keep the lock for kill_patience frames (others visible -> likely dead)
        or miss_patience frames (empty frame); only then drop and re-acquire / approach.

    Vision-only: 'killed' is inferred from the box staying gone; we never read game truth.
    """

    def __init__(self, match_frac=0.045, size_ratio=2.0, kill_patience=4, miss_patience=6,
                 acquire_fov=ACQUIRE_FOV_DEG, retain_fov=RETAIN_FOV_DEG,
                 switch_margin=SWITCH_MARGIN_DEG, switch_cooldown=SWITCH_COOLDOWN):
        self.match_frac = match_frac          # MIN identity-match radius as a fraction of frame width
        self.size_ratio = size_ratio          # reject an identity-match whose area differs by more than this
        self.kill_patience = kill_patience    # missing frames (others visible) -> declare killed
        self.miss_patience = miss_patience    # missing frames (empty frame) -> drop the lock
        self.acquire_fov = acquire_fov        # cone (deg) to LOCK/FIRE on a target ("engage" zone)
        self.retain_fov = retain_fov          # cone (deg) to KEEP a locked target (wider = sticky)
        self.switch_margin = switch_margin    # deg a challenger must beat the lock by to steal it
        self.switch_cooldown = switch_cooldown  # min frames between switches
        self._last = None                     # committed Detection (last matched box)
        self._missing = 0                     # consecutive unmatched frames
        self._since_commit = 0                # frames since the last acquire/switch (switch cooldown)
        self.engaging = False                 # True = locked target is IN the fire cone (aim+fire);
                                              # False = approaching an off-cone target (turn gently, hold fire)

    def _lock(self, det):
        """Stay on the SAME committed target (update its box, reset the miss counter)."""
        self._last = det
        self._missing = 0
        return det

    def _commit(self, det):
        """Acquire / switch to a NEW target (also resets the switch cooldown). None passes through."""
        if det is None:
            return None
        self._since_commit = 0
        return self._lock(det)

    def _matches(self, d, frame_w, frame_h):
        """Is d plausibly the SAME committed target? An ANISOTROPIC position gate plus a size-continuity
        check. The arena spread and the bot's pan are HORIZONTAL while vertical motion is ~0, and a zombie
        box is ~3.3x taller than wide — so a height-based radius would be far too wide horizontally and
        match an adjacent-cell respawn after a kill. Gate the horizontal tolerance by WIDTH (tight, rejects
        horizontal neighbours) and the vertical by height (loose, tolerates bbox jitter)."""
        last = self._last
        rx = max(2.0 * last.w, self.match_frac * frame_w)    # horizontal: tight
        ry = max(1.5 * last.h, 0.05 * frame_h)               # vertical: loose (no real vertical motion)
        if abs(d.cx - last.cx) > rx or abs(d.cy - last.cy) > ry:
            return False
        a0, a1 = last.area, d.area
        return a1 <= self.size_ratio * a0 and a0 <= self.size_ratio * a1

    def _acquire_or_approach(self, prep):
        """Pick a target to aim at and set self.engaging. Prefer the crosshair-nearest one INSIDE the fire
        cone (engaging=True -> aim + fire). If none is in-cone but zombies are visible elsewhere, return the
        crosshair-nearest OFF-cone one (engaging=False) so the bot turns toward it — this is how it 'looks
        to the other side' after clearing the targets in front, instead of idling with zombies on screen."""
        incone = _argmin_in_prep(prep, self.acquire_fov)
        if incone is not None:
            self.engaging = True
            return self._commit(incone)
        approach = _argmin_in_prep(prep, 180.0)   # nearest crosshair at ANY angle
        self.engaging = False
        return self._commit(approach) if approach is not None else None

    def select(self, dets, frame_w, frame_h, fov_deg=SEL_FOV_DEG):
        """dets: already conf-filtered Detections. Returns the target to aim at, or None (command nothing).
        Read self.engaging after: True = fire-ready (in cone), False = approaching an off-cone target."""
        self.engaging = False
        # Cache (det, d_yaw, d_pitch, ang_off, sel_cost) once per call -> 3-5x fewer atan2/sqrt calls vs
        # recomputing inside every helper (_ang_off, _sel_cost, _argmin_in_cone).
        prep = _prep_bearings(dets, frame_w, frame_h, fov_deg) if dets else []

        # no lock -> acquire (in-cone) or approach (off-cone)
        if self._last is None:
            return self._acquire_or_approach(prep)

        self._since_commit += 1

        # is the committed target present this frame? (identity gate). Find by spatial proximity, then verify.
        matched = None
        matched_tuple = None
        if prep:
            matched_tuple = min(prep, key=lambda t: (t[0].cx - self._last.cx) ** 2 + (t[0].cy - self._last.cy) ** 2)
            if self._matches(matched_tuple[0], frame_w, frame_h):
                matched = matched_tuple[0]
            else:
                matched_tuple = None

        if matched is not None:
            off = matched_tuple[3]   # ang_off from the cache (no extra atan2)
            # drifted past even the (wider) retain cone -> re-pick (in-cone engage, else approach)
            if off > self.retain_fov:
                self._last = None
                return self._acquire_or_approach(prep)
            # switch-hysteresis: a clearly-better IN-CONE challenger steals the lock once the cooldown has
            # elapsed -- OR immediately if we're only APPROACHING (matched is still off-cone): a far target
            # we're merely panning toward must not earn hysteresis protection against a target in the fire cone.
            if off > self.acquire_fov or self._since_commit >= self.switch_cooldown:
                best = _argmin_in_prep(prep, self.acquire_fov, exclude=matched)
                if best is not None:
                    # find best's cached cost in one pass (small N, negligible) and compare
                    best_cost = next(t[4] for t in prep if t[0] is best)
                    matched_cost = matched_tuple[4]
                    if best_cost + self.switch_margin < matched_cost:
                        self.engaging = True
                        return self._commit(best)
            # stay on the committed target; fire-ready only while it is inside the fire cone
            self.engaging = off <= self.acquire_fov
            return self._lock(matched)

        # committed target missing this frame -> coast (command nothing), keep the lock
        self._missing += 1
        patience = self.kill_patience if dets else self.miss_patience
        if self._missing < patience:
            return None
        # gone long enough -> drop and re-acquire (in-cone engage, else approach the other side)
        self._last = None
        self._missing = 0
        return self._acquire_or_approach(prep)
