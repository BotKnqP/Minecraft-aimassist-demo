"""Unit tests for the v0.4 high-freq bearing tracker.

  python -m mc_bow_agent.selftest_bearing
"""
from .bearing_tracker import (TargetState, expected_mod_turn,
                              MOD_TURN_GAIN, MOD_MOVE_DEADZONE_DEG, MOD_LIVE_MAX_STEP_DEG)


def approx(a, b, t=1e-6):
    return abs(a - b) <= t


def test_expected_mod_turn_deadzone():
    """Below the mod's MOVE_DEADZONE (1.2 deg) the mod commands zero turn."""
    assert expected_mod_turn(0.5) == 0.0
    assert expected_mod_turn(-1.1) == 0.0
    assert expected_mod_turn(1.2 - 0.001) == 0.0


def test_expected_mod_turn_gain_and_clamp():
    """Above deadzone the mod applies TURN_GAIN with clamp to ±LIVE_MAX_STEP_DEG."""
    assert approx(expected_mod_turn(5.0), 5.0 * MOD_TURN_GAIN)
    assert approx(expected_mod_turn(-8.0), -8.0 * MOD_TURN_GAIN)
    # large commanded turn -> clamped at ±LIVE_MAX_STEP_DEG (10), so 30*0.45=13.5 -> 10
    assert approx(expected_mod_turn(30.0), MOD_LIVE_MAX_STEP_DEG)
    assert approx(expected_mod_turn(-30.0), -MOD_LIVE_MAX_STEP_DEG)


def test_has_target_starts_false_and_becomes_true_on_measurement():
    s = TargetState(max_predict_ms=300)
    assert not s.has_target(now_ms=0.0)
    s.on_measurement(d_yaw=5.0, d_pitch=-2.0, bbox_h=40, bbox_w=20, conf=0.8, now_ms=1000.0)
    assert s.has_target(now_ms=1000.0)
    assert s.has_target(now_ms=1299.0)        # still inside the 300 ms window
    assert not s.has_target(now_ms=1301.0)    # past the window


def test_current_bearing_returns_measurement_until_decayed_by_send():
    s = TargetState()
    s.on_measurement(d_yaw=5.0, d_pitch=-2.0, bbox_h=40, bbox_w=20, conf=0.8, now_ms=0.0)
    dy, dp, h, w, c = s.current_bearing()
    assert approx(dy, 5.0) and approx(dp, -2.0)
    assert approx(h, 40) and approx(w, 20) and approx(c, 0.8)


def test_on_send_subtracts_expected_mod_turn_from_held_bearing():
    """After commanding +5° yaw, the mod will turn 5*0.45=2.25° right; the held bearing must
    drop by 2.25° so the next emit is the residual error, not a duplicate of the original."""
    s = TargetState()
    s.on_measurement(d_yaw=5.0, d_pitch=-2.0, bbox_h=40, bbox_w=20, conf=0.8, now_ms=0.0)
    s.on_send(sent_d_yaw=5.0, sent_d_pitch=-2.0)   # mod actually turns 2.25 yaw and ZERO pitch (deadzone)
    dy, dp, _, _, _ = s.current_bearing()
    assert approx(dy, 5.0 - 5.0 * MOD_TURN_GAIN, 1e-6)
    # pitch was -2.0 which is in deadzone (|.| < 1.2)? Actually 2.0 > 1.2, so it IS above the deadzone
    assert approx(dp, -2.0 - (-2.0) * MOD_TURN_GAIN, 1e-6)


def test_repeated_sends_converge_to_zero_for_static_target():
    """A static target + repeated 'aim & send' converges to bearing ~0 in a few iterations — i.e. no
    second/third-shot drift. This is the core property v0.4 buys vs the lock-step version."""
    s = TargetState()
    s.on_measurement(d_yaw=10.0, d_pitch=0.0, bbox_h=40, bbox_w=20, conf=0.8, now_ms=0.0)
    history = []
    for _ in range(8):
        dy, _, _, _, _ = s.current_bearing()
        history.append(dy)
        s.on_send(sent_d_yaw=dy, sent_d_pitch=0.0)
    # bearing strictly contracts toward zero (geometric decay at rate 1-gain = 0.55)
    for i in range(1, len(history)):
        assert abs(history[i]) <= abs(history[i - 1]), f"non-contractive: {history}"
    # converges within deadzone in a handful of ticks
    assert abs(history[-1]) < MOD_MOVE_DEADZONE_DEG, f"didn't converge: {history}"


def test_age_ms_and_reset():
    s = TargetState(max_predict_ms=300)
    s.on_measurement(d_yaw=1.0, d_pitch=0.0, bbox_h=40, bbox_w=20, conf=0.8, now_ms=100.0)
    assert approx(s.age_ms(now_ms=150.0), 50.0)
    s.reset()
    assert not s.has_target(now_ms=150.0)


def test_predict_window_drops_target_after_grace():
    """Per GPT's acceptance criteria: 300 ms after last measurement, has_target falls to False."""
    s = TargetState(max_predict_ms=300)
    s.on_measurement(d_yaw=0.0, d_pitch=0.0, bbox_h=40, bbox_w=20, conf=0.8, now_ms=0.0)
    assert s.has_target(now_ms=299.0)
    assert s.has_target(now_ms=300.0)
    assert not s.has_target(now_ms=301.0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL {len(tests)} BEARING TESTS PASSED")


if __name__ == "__main__":
    main()
