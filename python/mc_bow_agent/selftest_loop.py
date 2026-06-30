"""Tests for the runtime socket loop WITHOUT a live game.

A mock "mod" server (in a thread) sends recorded PNG frames and collects the
actions the client sends back. Two deterministic tests (protocol round-trip +
loop with a fake detector); plus real_e2e() to run the actual YOLO over the
socket on a recorded frame.

  python -m mc_bow_agent.selftest_loop
  python -c "from mc_bow_agent.selftest_loop import real_e2e; real_e2e('<best.pt>','<frame.png>')"
"""
import glob
import socket
import threading

from . import protocol as P
from .aim import Detection
from .runtime_loop import run_client


def approx(a, b, t=1e-6):
    return abs(a - b) <= t


def test_protocol_raw_frame_roundtrip():
    """Legacy 'R' raw-BGR payload [magic 'R'][W u16 BE][H u16 BE][W*H*3 BGR] decodes losslessly.
    decode_frame returns (frame, meta_dict); for 'R' the meta is empty."""
    import numpy as np
    W, H = 17, 11
    rng = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    payload = bytes([ord('R'), (W >> 8) & 0xff, W & 0xff, (H >> 8) & 0xff, H & 0xff]) + rng.tobytes()
    out, meta = P.decode_frame(payload)
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rng)
    assert meta == {}


def test_protocol_raw_w_frame_carries_fov():
    """The current 'W' raw frame ships capture_ms AND the FOV the projection matrix used. Python uses
    the per-frame FOV (not a hard-coded 70) so a user running 93° gets correct bearings + range."""
    import numpy as np
    W, H = 5, 7
    cap_ms = 1234567890
    fov_x100 = 9300                                    # 93.00°
    rng = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    hdr = bytes([ord('W'), (W >> 8) & 0xff, W & 0xff, (H >> 8) & 0xff, H & 0xff])
    cap = cap_ms.to_bytes(8, "big")
    fov = fov_x100.to_bytes(2, "big")
    payload = hdr + cap + fov + rng.tobytes()
    out, meta = P.decode_frame(payload)
    assert out.shape == (H, W, 3) and np.array_equal(out, rng)
    assert meta["capture_ms"] == cap_ms
    assert approx(meta["fov_deg"], 93.0, 1e-6)
    # garbage FOV (0 or out of [10, 175]) -> meta omits fov_deg, caller falls back to CLI
    bad = hdr + cap + (0).to_bytes(2, "big") + rng.tobytes()
    _, meta_bad = P.decode_frame(bad)
    assert "fov_deg" not in meta_bad


def test_protocol_raw_v_frame_carries_capture_ms():
    """The new 'V' raw-versioned payload carries an 8-byte capture_unix_ms so Python can back-correct
    a measurement by the mod-turn that happened during the inference round-trip."""
    import numpy as np
    W, H = 17, 11
    cap_ms = 0x0123456789ABCDEF
    rng = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    hdr = bytes([ord('V'), (W >> 8) & 0xff, W & 0xff, (H >> 8) & 0xff, H & 0xff])
    cap = cap_ms.to_bytes(8, "big")
    payload = hdr + cap + rng.tobytes()
    out, meta = P.decode_frame(payload)
    assert out.shape == (H, W, 3)
    assert np.array_equal(out, rng)
    assert meta == {"capture_ms": cap_ms}


def test_protocol_binary_action_roundtrip():
    """encode_action_bin -> decode_action round-trips exactly. The mod sniffs the leading byte ('A' for
    binary, '{' for JSON) so both encodings can flow over the same socket interchangeably."""
    act = {"has_target": True, "d_yaw": 1.5, "d_pitch": -2.25, "range": 12.5, "fire_ok": True,
           "n_det": 3, "boxes": [[10, 20, 30, 40, 1], [-5, 0, 5, 10, 2]]}
    enc = P.encode_action_bin(act)
    assert enc[0] == 0x41                              # magic 'A'
    dec = P.decode_action(enc)
    assert dec["has_target"] is True
    assert approx(dec["d_yaw"], 1.5, 1e-4)
    assert approx(dec["d_pitch"], -2.25, 1e-4)
    assert approx(dec["range"], 12.5, 1e-4)
    assert dec["fire_ok"] is True
    assert dec["n_det"] == 3
    assert dec["boxes"] == act["boxes"]
    # round-trip the JSON branch through the same decoder too
    j = P.encode_action(act)
    assert chr(j[0]) == "{"
    dec2 = P.decode_action(j)
    assert dec2["has_target"] is True and dec2["boxes"] == act["boxes"]


def test_protocol_raw_frame_size_mismatch_raises():
    """A truncated raw payload must fail loudly, not silently produce a garbage frame."""
    import numpy as np
    payload = bytes([ord('R'), 0, 4, 0, 4]) + (b"\x00" * 7)   # claims 4x4 BGR = 48B but only 7B given
    try:
        P.decode_frame(payload)
    except ValueError as e:
        assert "size mismatch" in str(e), e
        return
    assert False, "expected ValueError on size mismatch"


class FakeDetector:
    """One big centred zombie regardless of frame content (deterministic)."""
    def detect(self, frame):
        h, w = frame.shape[:2]
        det = Detection.from_xyxy(w / 2 - 30, h / 2 - 60, w / 2 + 30, h / 2 + 60, 0.9)  # h=120
        return [det], (h, w)


def _sample_frame_bytes():
    cands = sorted(glob.glob(r"D:\projects\mc-bow-agent\runs\run_*\frames\frame_*.png"))
    if not cands:
        raise SystemExit("no recorded frame found for the mock test")
    with open(cands[len(cands) // 2], "rb") as f:
        return f.read()


def _mock_mod_server(frame_bytes, n, received):
    """Returns (srv_socket, port, thread). Server sends n frames, collects n actions."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(5.0)
    port = srv.getsockname()[1]

    def serve():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            return
        try:
            for _ in range(n):
                P.send_msg(conn, frame_bytes)
                a = P.recv_msg(conn)
                if a is None:
                    break
                received.append(P.decode_action(a))
        finally:
            conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    return srv, port, th


def test_protocol_roundtrip():
    a, b = socket.socketpair()
    try:
        P.send_msg(a, b"hello world")
        assert P.recv_msg(b) == b"hello world"
        act = {"has_target": True, "d_yaw": 1.5, "fire_ok": False}
        P.send_msg(b, P.encode_action(act))
        assert P.decode_action(P.recv_msg(a)) == act
    finally:
        a.close()
        b.close()


def test_loop_with_mock_mod():
    n = 3
    received = []
    srv, port, th = _mock_mod_server(_sample_frame_bytes(), n, received)
    cli = socket.create_connection(("127.0.0.1", port))
    try:
        frames, targets = run_client(FakeDetector(), cli, max_frames=n)
    finally:
        cli.close()
        srv.close()
    th.join(timeout=5)

    assert frames == n and targets == n, (frames, targets)
    assert len(received) == n, received
    for act in received:
        assert act["has_target"] is True
        assert approx(act["d_yaw"], 0.0, 1e-3)      # centred -> no yaw turn
        assert act["d_pitch"] < 0                    # aims up for arrow drop
        assert act["range"] > 0 and act["n_det"] == 1
    print("  sample action:", received[0])


def real_e2e(weights, frame_path, n=3, device="cpu", conf=0.5):
    """Run the ACTUAL detector over the socket on a recorded frame; print actions."""
    from .runtime import Detector
    with open(frame_path, "rb") as f:
        fb = f.read()
    received = []
    srv, port, th = _mock_mod_server(fb, n, received)
    det = Detector(weights, conf=conf, device=device)
    cli = socket.create_connection(("127.0.0.1", port))
    try:
        run_client(det, cli, conf=conf, max_frames=n)
    finally:
        cli.close()
        srv.close()
    th.join(timeout=10)
    for a in received:
        print("action:", a)
    return received


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL {len(tests)} LOOP TESTS PASSED")


if __name__ == "__main__":
    main()
