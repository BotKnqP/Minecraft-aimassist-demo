"""Lightweight per-frame detection smoother.

The weak ~0.39 mAP detector flickers on far zombies: a true positive appears in frame N, vanishes in
N+1, returns in N+2. The in-game ESP boxes (and downstream aim) inherit that jitter. This module is
a tiny IoU tracker that:

  * associates each new detection to the nearest previous TRACK by IoU (greedy);
  * EMA-smooths the box coords so the visible rectangle doesn't twitch;
  * KEEPS a track alive for `max_miss` consecutive missing frames (hold-through-miss), so a single
    dropped frame doesn't visually erase a zombie that's clearly still there;
  * DOES NOT filter brand-new tracks on hit count by default (no latency on first acquire) — single-
    frame false positives are still surfaced, but `min_hits > 1` suppresses them too if needed.

This sits between Detector.detect() and downstream consumers (TargetTracker / aim / ESP overlay).
It is class-agnostic and stateless across socket reconnects (DetectionSmoother is created per
run_client call, so a reset just gives you a fresh tracker).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .aim import Detection


def iou_xywh(a: Detection, b: Detection) -> float:
    """IoU between two Detections (axis-aligned bboxes, centre+wh layout)."""
    ax0 = a.cx - a.w / 2.0
    ay0 = a.cy - a.h / 2.0
    ax1 = a.cx + a.w / 2.0
    ay1 = a.cy + a.h / 2.0
    bx0 = b.cx - b.w / 2.0
    by0 = b.cy - b.h / 2.0
    bx1 = b.cx + b.w / 2.0
    by1 = b.cy + b.h / 2.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 1e-9 else 0.0


@dataclass
class _Track:
    det: Detection          # current (EMA-smoothed) detection
    miss: int               # consecutive frames since the last successful match
    age: int                # frames since track creation (a hit counter)


class DetectionSmoother:
    """Frame-to-frame stabiliser for the detector output. Cheap, no model change required.

    Tuning:
      iou_thresh : 0.3        # below this, a new det is considered a DIFFERENT object
      max_miss   : 2          # frames a missing track lingers (~200 ms at 10 fps)
      ema        : 0.7        # 1.0 = no smoothing, 0.0 = freeze the box
      min_hits   : 1          # 1 = surface tracks immediately on first sighting (low latency);
                              # 2+ = suppress single-frame false positives (one frame of latency).
    """

    def __init__(self, iou_thresh: float = 0.3, max_miss: int = 2,
                 ema: float = 0.7, min_hits: int = 1):
        self.iou_thresh = float(iou_thresh)
        self.max_miss = int(max_miss)
        self.ema = float(ema)
        self.min_hits = int(min_hits)
        self._tracks: List[_Track] = []

    def reset(self) -> None:
        self._tracks.clear()

    def _smooth(self, old: Detection, new: Detection) -> Detection:
        """EMA-blend the OLD track's coords toward the NEW detection. Conf takes the new value."""
        a = self.ema
        return Detection(
            cx=a * new.cx + (1.0 - a) * old.cx,
            cy=a * new.cy + (1.0 - a) * old.cy,
            w=a * new.w + (1.0 - a) * old.w,
            h=a * new.h + (1.0 - a) * old.h,
            conf=new.conf,
        )

    def update(self, dets: List[Detection]) -> List[Detection]:
        """Run one frame through the smoother. Returns the stabilised list (matched + held)."""
        n_tracks = len(self._tracks)
        n_dets = len(dets)

        # 1) Greedy IoU matching: build all candidate pairs above the threshold, sort by IoU desc,
        # then assign — each track and each det used at most once. This is BYTETrack's matching
        # cascade in spirit, simplified to a single pass since we don't differentiate high/low conf.
        cand = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(dets):
                iou = iou_xywh(t.det, d)
                if iou >= self.iou_thresh:
                    cand.append((iou, ti, di))
        cand.sort(reverse=True)

        track_used = [False] * n_tracks
        det_used = [False] * n_dets
        for _iou, ti, di in cand:
            if not track_used[ti] and not det_used[di]:
                t = self._tracks[ti]
                t.det = self._smooth(t.det, dets[di])
                t.miss = 0
                t.age += 1
                track_used[ti] = True
                det_used[di] = True

        # 2) Unmatched detections -> new tentative tracks
        for di in range(n_dets):
            if not det_used[di]:
                self._tracks.append(_Track(det=dets[di], miss=0, age=1))

        # 3) Unmatched tracks -> bump miss counter
        for ti in range(n_tracks):
            if not track_used[ti]:
                self._tracks[ti].miss += 1

        # 4) Drop tracks that have been missing too long
        self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]

        # 5) Surface tracks that have met min_hits AND are alive (miss == 0 OR holding)
        return [t.det for t in self._tracks if t.age >= self.min_hits]
