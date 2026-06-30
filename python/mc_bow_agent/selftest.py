"""Dependency-light self test (numpy only).

Run from the `python/` dir:   python -m mc_bow_agent.selftest
"""
import numpy as np

from . import constants as C
from .action_mapping import (CameraQuantizer, bin_centers_deg, aim_angles,
                             relative_camera_delta, world_to_camera_action,
                             bow_power, arrow_speed, is_fireable)
from .ballistic import height_at_distance, solve_pitch


def approx(a, b, tol=1e-2):
    return abs(a - b) <= tol


def test_camera_bins():
    q = CameraQuantizer()
    assert np.allclose(bin_centers_deg(q), np.array(C.CAMERA_BIN_CENTERS_DEG), atol=0.01)
    assert int(q.discretize(0.0)) == C.CAMERA_ZERO_BIN
    assert int(q.discretize(10)) == C.CAMERA_N_BINS - 1
    assert int(q.discretize(-10)) == 0
    assert int(q.discretize(0.2)) == C.CAMERA_ZERO_BIN     # tiny delta -> no turn
    assert int(q.discretize(99)) == C.CAMERA_N_BINS - 1    # clipped beyond +-10


def test_aim_angles():
    eye = (0.0, 1.62, 0.0)
    y, p = aim_angles(eye, (0.0, 1.62, 10.0))           # straight ahead, level
    assert approx(y, 0.0) and approx(p, 0.0), (y, p)
    y, p = aim_angles(eye, (10.0, 1.62, 0.0))           # +X (east) -> yaw -90
    assert approx(y, -90.0), y
    y, p = aim_angles(eye, (0.0, 1.62 - 10.0, 10.0))    # below -> pitch DOWN-positive
    assert p > 0 and approx(p, 45.0), p


def test_relative_delta():
    _, d_yaw = relative_camera_delta(170.0, 0.0, -170.0, 0.0)   # wrap +-180
    assert approx(d_yaw, 10.0), d_yaw                            # +20 wrapped, clip +10
    d_pitch, _ = relative_camera_delta(0.0, 0.0, 0.0, 40.0)     # clip big turn
    assert approx(d_pitch, 10.0), d_pitch


def test_bow():
    assert approx(bow_power(20), 1.0)
    assert approx(arrow_speed(20), 3.0)
    assert not is_fireable(2)
    assert is_fireable(3)


def test_pipeline():
    out = world_to_camera_action((0.0, 1.62, 0.0), (5.0, 1.62, 10.0), 0.0, 0.0)
    assert out["target_yaw"] < 0                          # target to the +X -> turn left
    assert -10 <= out["camera_deg"][1] <= 10


def test_ballistic():
    assert (height_at_distance(3.0, 0.0, 12.0) or 0) < 0.0   # flat shot drops
    pitch = solve_pitch(3.0, 12.0, 0.0)
    h = height_at_distance(3.0, pitch, 12.0)
    assert h is not None and abs(h) < 0.3, (pitch, h)
    assert pitch < 0.0, pitch                            # aim up = negative pitch


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
