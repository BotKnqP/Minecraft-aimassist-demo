"""v1 real-time loop: connect to the mod, then per tick recv frame -> detect ->
aim at nearest zombie -> send action. Lock-step (one action per frame).

  python -m mc_bow_agent.runtime_loop --weights runs/detect/mcbow_zombie_v2/weights/best.onnx \
      --host 127.0.0.1 --port 5555 --device cuda:0

The mod side (Java socket endpoint, TODO) sends frames and applies each action via
BowMacroController. Action JSON (see aim_to_action):
  {has_target, d_yaw, d_pitch, range, fire_ok, n_det}
Mod contract per tick: if has_target -> step view toward (cur_yaw+d_yaw,
cur_pitch+d_pitch) clamped <=10 deg/tick, hold the bow while fire_ok, release
(fire) when |d_yaw|,|d_pitch| are within ~2 deg AND draw is full (>=20 ticks);
else release the bow (optionally slow-scan). See docs/RUNTIME_PROTOCOL.md.
"""
import argparse
import os
import socket
import time
from dataclasses import replace

# Ease CUDA fragmentation when sharing the GPU with Minecraft. NOTE: expandable_segments is a Linux-only
# allocator feature — PyTorch ignores it on Windows, where the real OOM mitigations are a smaller --imgsz and
# the automatic CPU fallback (see Detector). Harmless to set on either OS; must be set BEFORE torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from . import protocol as P
from .aim import APPROACH_STEP_DEG, TargetTracker, aim_at
from .runtime import DEFAULT_FOV, DEFAULT_K


def aim_to_action(sol, n_det=0) -> dict:
    """AimSolution (or None) -> action dict sent to the mod."""
    if sol is None:
        return {"has_target": False, "d_yaw": 0.0, "d_pitch": 0.0,
                "range": -1.0, "fire_ok": False, "n_det": n_det}
    return {"has_target": True, "d_yaw": round(sol.d_yaw, 3),
            "d_pitch": round(sol.d_pitch, 3), "range": round(sol.range_blocks, 2),
            "fire_ok": bool(sol.fireable), "n_det": n_det}


def boxes_payload(dets, target, engaging):
    """Detection boxes for the in-game ESP overlay: [x0,y0,x1,y1,role] in the captured-frame (scaled-GUI)
    coords, which map 1:1 to the mod's HUD. role: 0=other(green), 1=engaged target(red), 2=approach(yellow)."""
    out = []
    for d in dets:
        role = (1 if engaging else 2) if d is target else 0
        out.append([int(round(d.cx - d.w / 2)), int(round(d.cy - d.h / 2)),
                    int(round(d.cx + d.w / 2)), int(round(d.cy + d.h / 2)), role])
    return out


def _show_detections(frame, dets, target, sol):
    """Live debug window: all boxes (green), the committed target (red), crosshair + aim info."""
    try:
        import cv2
    except ImportError:
        return
    img = frame.copy()
    for d in dets:
        x0, y0 = int(d.cx - d.w / 2), int(d.cy - d.h / 2)
        x1, y1 = int(d.cx + d.w / 2), int(d.cy + d.h / 2)
        sel = target is not None and d is target
        color = (0, 0, 255) if sel else (0, 200, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2 if sel else 1)
        cv2.putText(img, f"{d.conf:.2f}", (x0, max(8, y0 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 12, 1)
    if sol is not None:
        cv2.putText(img, f"R={sol.range_blocks:.1f} yaw{sol.d_yaw:+.0f} pit{sol.d_pitch:+.0f} fire={sol.fireable}",
                    (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("mcbow detections", cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(1)


def run_client(detector, sock, k=DEFAULT_K, fov=DEFAULT_FOV, conf=0.5,
               max_frames=None, on_step=None, debug=False, show=False, tracker=None):
    """Drive the lock-step loop on an open socket. `detector` is any object with
    detect(frame)->(detections, shape). Returns (frames, targets). Raises on a
    socket abort/reset (caught by the caller's reconnect loop)."""
    if tracker is None:
        tracker = TargetTracker()
    frames = targets = 0
    while max_frames is None or frames < max_frames:
        png = P.recv_msg(sock)              # socket error -> propagate -> caller reconnects
        if png is None:
            break  # mod closed the connection (clean EOF)
        if debug:
            print(f"[recv] frame_seq={frames} png_bytes={len(png)}")

        # Perception/aim is wrapped: a transient inference hiccup (e.g. cv2/torch OOM when
        # commit memory spikes) must NOT crash the program. On error we skip the frame and
        # still reply with a safe 'no target' so the mod's lock-step read doesn't hang.
        frame = None
        dets, target, sol = [], None, None
        try:
            frame = P.decode_frame(png)
            if frame is not None:
                h, w = frame.shape[:2]
                dets = detector.detect(frame)[0]
                target = tracker.select(dets, w, h, fov)  # aimbot lock: crosshair-nearest in FOV cone
                sol = aim_at(target, w, h, fov, k=k) if target is not None else None
                if sol is not None and not tracker.engaging:
                    # approaching an OFF-cone target ("look to the other side"): pan gently toward it and
                    # HOLD fire until it enters the cone (engaging) and the mod aligns within ~2 deg.
                    s = APPROACH_STEP_DEG
                    sol = replace(sol, fireable=False,
                                  d_yaw=max(-s, min(s, sol.d_yaw)),
                                  d_pitch=max(-s, min(s, sol.d_pitch)))
        except Exception as e:
            print(f"[frame {frames}] inference skipped ({type(e).__name__}: {e})")
            dets, target, sol = [], None, None

        action = aim_to_action(sol, n_det=len(dets))
        action["boxes"] = boxes_payload(dets, target, getattr(tracker, "engaging", False))
        P.send_msg(sock, P.encode_action(action))       # socket error -> propagate -> reconnect
        if debug:
            print(f"[send] d_yaw={action['d_yaw']} d_pitch={action['d_pitch']} "
                  f"range={action['range']} fire_ok={action['fire_ok']}")
        if show and frame is not None:
            try:
                _show_detections(frame, dets, target, sol)
            except Exception:
                pass   # a display error must never crash/exit the control loop
        frames += 1
        targets += 1 if sol is not None else 0
        if on_step is not None:
            on_step(frames, len(dets), sol, action)
    return frames, targets


def main(argv=None):
    ap = argparse.ArgumentParser(description="v1 real-time vision->bow loop (mod client).")
    ap.add_argument("--weights", required=True, help="best.pt or best.onnx")
    ap.add_argument("--host", default=P.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=P.DEFAULT_PORT)
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--fov", type=float, default=DEFAULT_FOV)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence; 0.25 catches far more zombies (the detector is "
                         "weak ~0.39 mAP); raise it if the bot fires at false positives")
    ap.add_argument("--device", default="cpu",
                    help="'cpu' recommended for the LIVE run (Minecraft already holds the GPU; "
                         "a CUDA context here fights it for VRAM/commit). 'cuda:0' only if you "
                         "freed GPU + commit headroom.")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference size; default auto = 640 on GPU (the trained size -> best recall) / "
                         "320 on CPU (speed). Pass e.g. 416 to override.")
    ap.add_argument("--debug-protocol", action="store_true",
                    help="print per-frame (seq, png bytes, shape) and per-action (d_yaw,d_pitch,range,fire_ok)")
    ap.add_argument("--show", action="store_true",
                    help="open an OpenCV window of the frame + detection boxes + committed target")
    a = ap.parse_args(argv)

    # Resolve the EFFECTIVE device FIRST (cuda only if actually available), THEN derive imgsz from it — else a
    # cuda-requested-but-unavailable run would execute the heavy 640 model on CPU (the pressure 320 avoids).
    device = a.device
    if str(device).startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("[runtime] CUDA requested but not available -> CPU")
                device = "cpu"
        except Exception:
            device = "cpu"
    imgsz = a.imgsz if a.imgsz is not None else (640 if str(device).startswith("cuda") else 320)
    print(f"[runtime] device={device} imgsz={imgsz} conf={a.conf}")

    from .runtime import Detector
    detector = Detector(a.weights, conf=a.conf, device=device, imgsz=imgsz)

    t0 = [time.time()]

    def status(frames, ndet, sol, action):
        if frames % 20 == 0:
            now = time.time()
            fps = 20.0 / max(now - t0[0], 1e-6)
            t0[0] = now
            tgt = (f"range={action['range']}blk fire={action['fire_ok']} "
                   f"d_yaw={action['d_yaw']:+.1f}") if action["has_target"] else "no target"
            print(f"[{frames}] {fps:4.1f} fps  det={ndet}  {tgt}")

    def connect():
        for _ in range(120):                # retry so start order doesn't matter
            try:
                return socket.create_connection((a.host, a.port), timeout=2)
            except OSError:
                time.sleep(1)
        return None

    try:
        while True:                          # reconnect loop: survive mod restarts / disconnects
            print(f"connecting to mod at {a.host}:{a.port} (press F7 in-game) ...")
            sock = connect()
            if sock is None:
                print("could not connect; retrying ...")
                continue
            print("connected. running (Ctrl-C to stop).")
            t0[0] = time.time()
            try:
                run_client(detector, sock, k=a.k, fov=a.fov, conf=a.conf,
                           on_step=status, debug=a.debug_protocol, show=a.show)
                print("mod disconnected; reconnecting ...")
            except Exception as e:           # ANY error -> reconnect, never crash (Ctrl-C still exits)
                print(f"disconnected ({type(e).__name__}: {e}); reconnecting ...")
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        if a.show:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
