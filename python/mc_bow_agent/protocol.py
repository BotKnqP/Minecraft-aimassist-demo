"""Wire protocol for the v1 runtime <-> mod socket bridge.

Lock-step over localhost TCP (mod = server, Python = client):
  each game tick the mod sends ONE frame, Python replies with ONE action.

Message framing (both directions): 4-byte big-endian uint32 length + payload.
  * mod -> Python : payload = PNG bytes of the (downscaled, HUD-stripped) frame
                    at the SAME resolution the detector was trained on (427xH).
  * Python -> mod : payload = UTF-8 JSON action (see encode_action).

Keeping the mod as the frame source (not Python-side mss capture) means the
frames Python sees match training exactly, so the detector + calibrated k apply
directly with no rescaling.
"""
import json
import struct

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555


def send_msg(sock, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_n(sock, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    """Read one length-prefixed message. Returns bytes, or None if the peer closed."""
    hdr = _recv_n(sock, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack(">I", hdr)
    return _recv_n(sock, n)


def encode_action(d: dict) -> bytes:
    """JSON action — kept as the human-readable default for tests / debug tooling."""
    return json.dumps(d).encode("utf-8")


# Binary action layout, big-endian:
#   B  magic (0x41 = 'A')
#   B  has_target
#   f  d_yaw            (4 bytes)
#   f  d_pitch          (4 bytes)
#   f  range            (4 bytes)
#   B  fire_ok
#   H  n_det
#   H  n_boxes          -> header total = 19 bytes
# Then n_boxes * (h h h h h)  (5 * int16 = 10 bytes each):  x0 y0 x1 y1 role
_BIN_ACTION_HEADER = ">BBfffBHH"
_BIN_BOX = ">hhhhh"


def encode_action_bin(d: dict) -> bytes:
    """Binary action — drops the JSON encode/parse + GC pressure on the mod-side reader. Magic byte 0x41
    ('A') distinguishes from JSON ('{' = 0x7B); the mod sniffs the first byte to route. Same field shape
    as encode_action; see _BIN_ACTION_HEADER for the layout."""
    boxes = d.get("boxes") or []
    head = struct.pack(_BIN_ACTION_HEADER,
                       0x41,
                       1 if d.get("has_target", False) else 0,
                       float(d.get("d_yaw", 0.0)),
                       float(d.get("d_pitch", 0.0)),
                       float(d.get("range", -1.0)),
                       1 if d.get("fire_ok", False) else 0,
                       int(d.get("n_det", 0)) & 0xffff,
                       len(boxes) & 0xffff)
    if not boxes:
        return head
    body = bytearray()
    for b in boxes:
        body += struct.pack(_BIN_BOX, int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]))
    return head + bytes(body)


def decode_action(buf: bytes) -> dict:
    """Decode either binary ('A' first byte) or JSON ('{' first byte) action to the same dict shape.
    Tests / tools can keep round-tripping via this without caring which encoding the live loop uses."""
    if buf and buf[0] == 0x41:                                   # 'A' — binary
        (_, ht, dy, dp, rng, fo, n_det, n_boxes) = struct.unpack_from(_BIN_ACTION_HEADER, buf, 0)
        off = struct.calcsize(_BIN_ACTION_HEADER)
        boxes = []
        for _i in range(n_boxes):
            boxes.append(list(struct.unpack_from(_BIN_BOX, buf, off)))
            off += struct.calcsize(_BIN_BOX)
        return {"has_target": bool(ht), "d_yaw": float(dy), "d_pitch": float(dp),
                "range": float(rng), "fire_ok": bool(fo), "n_det": int(n_det), "boxes": boxes}
    return json.loads(buf.decode("utf-8"))


def decode_frame(buf: bytes):
    """Decode a frame payload to (HxWx3 BGR ndarray, meta dict).

    Three wire formats, distinguished by the first byte:
      * 'V' (0x56) — VERSIONED raw BGR, fast path WITH capture timestamp:
          [magic 'V'][W:u16 BE][H:u16 BE][capture_unix_ms:u64 BE][BGR bytes...]
          meta = {'capture_ms': <int>}  (mod's System.currentTimeMillis at GL readback)
      * 'R' (0x52) — legacy raw BGR (no timestamp): [magic 'R'][W:u16 BE][H:u16 BE][BGR bytes...]
          meta = {}
      * 0x89 — PNG: decoded via cv2.imdecode (or PIL fallback). meta = {}
    """
    import numpy as np
    if buf and buf[0] == 0x56:                                   # 'V' — versioned raw BGR (+ capture_ms)
        w = int.from_bytes(buf[1:3], "big")
        h = int.from_bytes(buf[3:5], "big")
        cap_ms = int.from_bytes(buf[5:13], "big")
        expected = 13 + w * h * 3
        if len(buf) != expected:
            raise ValueError(f"raw-v frame size mismatch: header says {w}x{h} (need {expected} bytes), got {len(buf)}")
        frame = np.frombuffer(buf, dtype=np.uint8, count=w * h * 3, offset=13).reshape(h, w, 3)
        return frame, {"capture_ms": cap_ms}
    if buf and buf[0] == 0x52:                                   # 'R' — legacy raw BGR
        w = int.from_bytes(buf[1:3], "big")
        h = int.from_bytes(buf[3:5], "big")
        expected = 5 + w * h * 3
        if len(buf) != expected:
            raise ValueError(f"raw frame size mismatch: header says {w}x{h} (need {expected} bytes), got {len(buf)}")
        frame = np.frombuffer(buf, dtype=np.uint8, count=w * h * 3, offset=5).reshape(h, w, 3)
        return frame, {}
    try:
        import cv2
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), {}
    except ImportError:
        import io
        from PIL import Image
        rgb = np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
        return rgb[:, :, ::-1].copy(), {}                        # RGB -> BGR
