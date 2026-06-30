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
    return json.dumps(d).encode("utf-8")


def decode_action(b: bytes) -> dict:
    return json.loads(b.decode("utf-8"))


def decode_frame(png_bytes: bytes):
    """PNG bytes -> HxWx3 BGR ndarray (matches how Ultralytics loads images, so
    train/infer colour order is consistent). Uses cv2 if present, else PIL."""
    import numpy as np
    try:
        import cv2
        return cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    except ImportError:
        import io
        from PIL import Image
        rgb = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR
