"""Unit tests for the DetectionSmoother.

  python -m mc_bow_agent.selftest_smoother
"""
from .aim import Detection
from .detect_tracker import DetectionSmoother, iou_xywh


def approx(a, b, t=1e-6):
    return abs(a - b) <= t


def _box(cx, cy, w, h, conf=0.9):
    return Detection(cx=cx, cy=cy, w=w, h=h, conf=conf)


def test_iou_xywh_basic():
    a = _box(50, 50, 20, 20)               # x: 40-60, y: 40-60
    assert approx(iou_xywh(a, a), 1.0)     # self-IoU = 1
    b = _box(50, 50, 10, 10)               # nested centred
    # inter = 100, union = 400 + 100 - 100 = 400 -> 0.25
    assert approx(iou_xywh(a, b), 100 / 400)
    far = _box(500, 500, 20, 20)
    assert iou_xywh(a, far) == 0.0


def test_holds_through_one_miss_frame():
    """A detection that vanishes for ONE frame must still be surfaced (hold-through-miss)."""
    sm = DetectionSmoother(iou_thresh=0.3, max_miss=2, ema=1.0, min_hits=1)
    d = _box(100, 100, 40, 60)
    out = sm.update([d])
    assert len(out) == 1                              # frame 1: appears
    out = sm.update([])                               # frame 2: detector drops it
    assert len(out) == 1                              # but the smoother still surfaces it
    assert approx(out[0].cx, 100) and approx(out[0].cy, 100)
    out = sm.update([d])                              # frame 3: reappears
    assert len(out) == 1


def test_drops_after_max_miss():
    """Past max_miss, the track is gone — no zombie box lingering forever."""
    sm = DetectionSmoother(iou_thresh=0.3, max_miss=2, ema=1.0, min_hits=1)
    d = _box(100, 100, 40, 60)
    sm.update([d])
    assert len(sm.update([])) == 1                    # miss 1 -> still alive
    assert len(sm.update([])) == 1                    # miss 2 -> last gasp
    assert len(sm.update([])) == 0                    # miss 3 -> dropped


def test_min_hits_suppresses_single_frame_false_positive():
    """min_hits=2 means a brand-new detection must survive 2 frames before it shows."""
    sm = DetectionSmoother(iou_thresh=0.3, max_miss=2, ema=1.0, min_hits=2)
    fp = _box(50, 50, 30, 40)
    assert len(sm.update([fp])) == 0                  # frame 1: tentative -> NOT surfaced
    assert len(sm.update([])) == 0                    # frame 2: missing -> still NOT surfaced (age stayed 1)
    # but a persistent detection eventually surfaces
    sm2 = DetectionSmoother(iou_thresh=0.3, max_miss=2, ema=1.0, min_hits=2)
    d = _box(100, 100, 40, 60)
    assert len(sm2.update([d])) == 0                  # frame 1: tentative
    out = sm2.update([d])                             # frame 2: age 2 -> surfaced
    assert len(out) == 1


def test_ema_smooths_box_jitter():
    """EMA<1 should pull the visible box toward the moving average instead of snapping."""
    sm = DetectionSmoother(iou_thresh=0.3, max_miss=2, ema=0.5, min_hits=1)
    sm.update([_box(100, 100, 40, 60)])
    out = sm.update([_box(110, 100, 40, 60)])         # jump of 10 px in x
    assert len(out) == 1
    # with ema=0.5, the smoothed cx is halfway between 100 (old) and 110 (new) = 105
    assert approx(out[0].cx, 105.0)


def test_no_match_two_distant_dets_yield_two_tracks():
    """Two well-separated boxes must NOT be associated to one track."""
    sm = DetectionSmoother(iou_thresh=0.3, max_miss=2, ema=1.0, min_hits=1)
    a = _box(100, 100, 40, 60)
    b = _box(400, 100, 40, 60)
    out = sm.update([a, b])
    assert len(out) == 2


def test_greedy_match_picks_highest_iou():
    """If two new dets overlap one track, the higher-IoU pair wins; the other becomes a new track."""
    sm = DetectionSmoother(iou_thresh=0.3, max_miss=2, ema=1.0, min_hits=1)
    sm.update([_box(100, 100, 40, 60)])
    # next frame: one almost-identical box (high IoU) + one shifted (lower IoU but still > 0.3 vs the SAME track)
    near = _box(102, 101, 40, 60)
    shifted = _box(120, 100, 40, 60)
    out = sm.update([near, shifted])
    # both must survive: `near` matched the track, `shifted` becomes its own track
    assert len(out) == 2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL {len(tests)} SMOOTHER TESTS PASSED")


if __name__ == "__main__":
    main()
