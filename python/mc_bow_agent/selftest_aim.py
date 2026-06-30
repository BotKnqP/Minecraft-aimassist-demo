"""Unit tests for the runtime aim logic (no deps beyond stdlib + this package).

  python -m mc_bow_agent.selftest_aim
"""
from .aim import (Detection, pick_nearest, focal_px, bearing_from_bbox,
                  range_from_bbox_height, aim_at, solve_from_detections, TargetTracker)


def approx(a, b, t=1e-6):
    return abs(a - b) <= t


def test_pick_nearest():
    d_small = Detection.from_xyxy(0, 0, 10, 10, 0.9)
    d_big = Detection.from_xyxy(0, 0, 50, 50, 0.8)
    d_lowconf = Detection.from_xyxy(0, 0, 100, 100, 0.3)
    assert pick_nearest([d_small, d_big, d_lowconf], 0.5) is d_big   # largest above conf
    assert pick_nearest([d_lowconf], 0.5) is None                    # all below conf
    assert pick_nearest([], 0.5) is None


def test_bearing():
    W, H, fov = 640, 360, 70.0
    f = focal_px(H, fov)
    y, p = bearing_from_bbox(W / 2, H / 2, W, H, fov)
    assert approx(y, 0) and approx(p, 0)                             # centred -> no turn
    y, _ = bearing_from_bbox(W / 2 + f, H / 2, W, H, fov)
    assert approx(y, 45.0, 1e-4)                                     # right -> +45 yaw
    _, p = bearing_from_bbox(W / 2, H / 2 + f, W, H, fov)
    assert approx(p, 45.0, 1e-4)                                     # below -> +45 pitch (down)
    y, _ = bearing_from_bbox(W / 2 - f, H / 2, W, H, fov)
    assert y < 0                                                     # left -> negative yaw


def test_range():
    assert approx(range_from_bbox_height(100, k=5000.0), 50.0)
    assert approx(range_from_bbox_height(250, k=5000.0), 20.0)
    assert range_from_bbox_height(0, k=5000.0) == float("inf")
    f = focal_px(360, 70.0)
    assert approx(range_from_bbox_height(f, 360, 70.0), 1.95, 1e-6)  # mob filling focal px -> 1.95 blocks


def test_aim_at():
    W, H, fov = 640, 360, 70.0
    det = Detection.from_xyxy(W / 2 - 20, H / 2 - 40, W / 2 + 20, H / 2 + 40, 0.9)  # centred, h=80
    sol = aim_at(det, W, H, fov, k=5000.0)
    assert approx(sol.d_yaw, 0, 1e-6)
    assert sol.drop_deg > 0 and sol.d_pitch < 0                      # aims UP to beat gravity
    assert approx(sol.range_blocks, 5000.0 / 80.0)
    assert sol.fireable == (sol.range_blocks <= 40.0)


def test_solve_from_detections():
    W, H, fov = 640, 360, 70.0
    dets = [Detection.from_xyxy(0, 0, 10, 10, 0.9),
            Detection.from_xyxy(W / 2 - 30, H / 2 - 30, W / 2 + 30, H / 2 + 30, 0.8)]
    sol = solve_from_detections(dets, W, H, fov, k=5000.0)
    assert sol is not None and sol.drop_deg > 0                      # picks the bigger (nearer) one
    assert solve_from_detections([], W, H, fov, k=5000.0) is None


def test_tracker_acquires_crosshair_nearest():
    """Lock the target NEAREST THE CROSSHAIR (engage), not the biggest box; an off-cone box is APPROACHED."""
    W, H, fov = 640, 360, 70.0
    tk = TargetTracker()
    center = Detection.from_xyxy(300, 160, 340, 200, 0.9)   # cx=320 -> ~0deg off-axis, small
    edge_big = Detection.from_xyxy(450, 120, 552, 240, 0.9)  # cx=501 -> ~35deg off-axis, BIG (in cone)
    assert tk.select([edge_big, center], W, H, fov) is center   # crosshair-nearest beats the bigger edge box
    assert tk.engaging is True                              # in the fire cone -> fire-ready

    tk2 = TargetTracker()
    far = Detection.from_xyxy(566, 160, 606, 200, 0.9)     # cx=586 -> ~46deg, beyond ACQUIRE_FOV(40)
    assert tk2.select([far], W, H, fov) is far             # off-cone -> APPROACH it (turn toward), not idle
    assert tk2.engaging is False                            # ...but not fire-ready until it enters the cone


def test_tracker_looks_to_other_side_after_clearing():
    """After the locked in-cone target vanishes, if the only zombies left are off to the side, turn toward
    the nearest (engaging=False) instead of idling -- the 'look to the other side' behavior."""
    W, H, fov = 640, 360, 70.0
    tk = TargetTracker(kill_patience=1)
    front = Detection.from_xyxy(300, 160, 340, 200, 0.9)   # centre, in cone -> locked + fire-ready
    assert tk.select([front], W, H, fov) is front and tk.engaging is True
    side = Detection.from_xyxy(560, 160, 600, 200, 0.9)    # cx=580 ~45deg off-cone, does NOT match `front`
    out = tk.select([side], W, H, fov)                     # front gone (kill_patience=1) -> approach the side
    assert out is side and tk.engaging is False


def test_tracker_no_whip_to_bigger_edge_box():
    """Once locked centre, a bigger box at the screen edge must NOT steal the aim (the reported bug)."""
    W, H, fov = 640, 360, 70.0
    tk = TargetTracker()
    center = Detection.from_xyxy(300, 160, 340, 200, 0.9)   # cx=320 ~0deg -> locked
    assert tk.select([center], W, H, fov) is center
    edge_big = Detection.from_xyxy(450, 120, 552, 240, 0.9)  # cx=501 ~35deg, much bigger
    assert tk.select([center, edge_big], W, H, fov) is center   # angular cost >> margin -> stay centred
    assert tk.select([center, edge_big], W, H, fov) is center


def test_tracker_switch_hysteresis():
    """A slightly-better challenger does NOT steal the lock; a clearly-better one does."""
    W, H, fov = 640, 360, 70.0
    tk = TargetTracker(switch_margin=8.0, switch_cooldown=0)
    A = Detection.from_xyxy(394, 160, 434, 200, 0.9)       # cx=414 -> ~20deg -> locked
    assert tk.select([A], W, H, fov) is A
    B = Detection.from_xyxy(364, 160, 404, 200, 0.9)       # cx=384 -> ~14deg, beats A by ~6deg (< margin 8)
    assert tk.select([A, B], W, H, fov) is A               # slightly-better challenger does NOT steal
    C = Detection.from_xyxy(322, 160, 362, 200, 0.9)       # cx=342 -> ~5deg, beats A by ~15deg (> margin 8)
    assert tk.select([A, C], W, H, fov) is C               # clearly-better challenger steals the lock


def test_tracker_miss_coast_and_reacquire():
    """Missing the locked target commands nothing (None) and keeps the lock, then re-acquires angularly."""
    W, H, fov = 640, 360, 70.0
    tk = TargetTracker(kill_patience=4, miss_patience=3)
    A = Detection.from_xyxy(300, 160, 340, 200, 0.9)       # cx=320 centre -> locked
    assert tk.select([A], W, H, fov) is A
    assert tk.select([], W, H, fov) is None                # empty frame -> coast (command nothing), keep lock
    assert tk.select([], W, H, fov) is None
    assert tk.select([], W, H, fov) is None                # miss_patience(3) reached -> drop; no dets -> None

    tk2 = TargetTracker(kill_patience=2)
    A2 = Detection.from_xyxy(394, 160, 434, 200, 0.9)      # cx=414 ~20deg -> locked
    B2 = Detection.from_xyxy(300, 160, 340, 200, 0.9)      # cx=320 centre, far enough to NOT match A2
    assert tk2.select([A2], W, H, fov) is A2
    assert tk2.select([B2], W, H, fov) is None             # A2 missing (B2 != A2) -> coast, miss #1
    assert tk2.select([B2], W, H, fov) is B2               # miss #2 == kill_patience -> re-acquire nearest = B2


def test_tracker_identity_horizontal_neighbor_rejected():
    """A tall-thin zombie's side neighbour (inside the OLD height-based radius but outside the new WIDTH-based
    one) must NOT be matched as the same target, so a kill cleanly re-acquires instead of teleporting the lock."""
    W, H, fov = 640, 360, 70.0
    tk = TargetTracker()
    A = Detection.from_xyxy(314, 150, 326, 210, 0.9)   # cx=320, w=12, h=60 (tall-thin)
    assert tk.select([A], W, H, fov) is A
    nb = Detection.from_xyxy(360, 150, 372, 210, 0.9)  # cx=366 (dx=46) = an adjacent-cell mob, same size
    # rx = max(2*12, 0.045*640) = 28.8 < 46, so nb is NOT 'still A' (old r=1.5*60=90 would have matched it)
    assert tk.select([nb], W, H, fov) is None          # treated as missing (coast), not a teleport onto nb


def test_tracker_identity_gate_rejects_bigger_box():
    """A much bigger box sitting where the target was must NOT be assumed to be the same target."""
    W, H, fov = 640, 360, 70.0
    tk = TargetTracker(kill_patience=3)
    A = Detection.from_xyxy(300, 160, 340, 200, 0.9)       # cx=320, area=1600 -> locked
    assert tk.select([A], W, H, fov) is A
    C = Detection.from_xyxy(260, 120, 380, 240, 0.9)       # same centre, area=14400 (9x) -> different mob
    assert tk.select([C], W, H, fov) is None               # size gate rejects -> treated as missing, not 'still A'


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL {len(tests)} AIM TESTS PASSED")


if __name__ == "__main__":
    main()
