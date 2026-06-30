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
import threading
import time
from dataclasses import replace

# Ease CUDA fragmentation when sharing the GPU with Minecraft. expandable_segments is a Linux-only allocator
# feature (PyTorch ignores it on Windows where it is a no-op), so gate it to Linux to keep the intent honest —
# on Windows the actual OOM mitigations are smaller --imgsz and the automatic CPU fallback in Detector.
import platform as _platform
if _platform.system() == "Linux":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from . import protocol as P
from .aim import APPROACH_STEP_DEG, TargetTracker, aim_at, aim_from_bearing, bearing_from_bbox
from .bearing_tracker import TargetState
from .detect_tracker import DetectionSmoother
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


class _LatestFrameRecv:
    """Background socket reader: pulls frames as fast as the mod sends them and exposes only the LATEST
    one to the main loop. Older frames silently drop. Pairs with the mod's pipelined writer (no lockstep)
    so detection wall-clock is bounded by inference time alone, not by capture+inference+send round-trip."""

    def __init__(self, sock):
        self.sock = sock
        self._lock = threading.Lock()
        self._latest = None       # most-recent frame payload, or None
        self._event = threading.Event()
        self._eof = False
        self._error = None
        self._stop = threading.Event()
        self._dropped = 0         # how many frames the main loop never saw (because a newer one arrived)
        self._thread = threading.Thread(target=self._run, name="mcbow-recv", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    payload = P.recv_msg(self.sock)
                except socket.timeout:
                    continue                          # the 2s socket timeout is a watchdog; just loop
                if payload is None:
                    self._eof = True
                    self._event.set()
                    return
                with self._lock:
                    if self._latest is not None:
                        self._dropped += 1            # we're about to overwrite -> the older frame is dropped
                    self._latest = payload
                self._event.set()
        except Exception as e:
            self._error = e
            self._event.set()

    def take(self, timeout=2.0):
        """Block up to `timeout` s for the next frame; return (payload, eof) and clear the slot.
        Buffered frames are delivered BEFORE a queued error/EOF is raised — otherwise a recv-thread crash
        right after pushing a frame would discard the last live data."""
        self._event.wait(timeout=timeout)
        with self._lock:
            payload = self._latest
            self._latest = None
            if payload is not None:
                self._event.clear()                   # more frames will re-set this
                return payload, False
            if self._error is not None:
                err = self._error
                self._error = None                    # one-shot raise
                raise err
            if self._eof:
                return None, True
            self._event.clear()                       # spurious wake / watchdog timeout
            return None, False

    def dropped(self):
        return self._dropped

    def close(self):
        self._stop.set()


def run_client(detector, sock, k=DEFAULT_K, fov=DEFAULT_FOV, conf=0.5,
               max_frames=None, on_step=None, debug=False, show=False, tracker=None,
               smoother=None, send_hz=20.0, predict_ms=300):
    """v0.4 async loop. Detection runs whenever a fresh frame arrives (10 Hz given the mod's
    capture interval); the SENDER ticks at `send_hz` (20 Hz) and emits the LATEST predicted
    bearing from TargetState every send. Between detections the bearing is decayed by the
    expected mod turn (gain + deadzone + clamp) so we don't overshoot. This turns the old
    'detect -> send -> wait' lock-step into 'detect on its own clock, send on its own clock'
    — control rate is no longer capped by detection rate. Lost-target grace = `predict_ms`."""
    if tracker is None:
        tracker = TargetTracker()
    recv = _LatestFrameRecv(sock)
    state = TargetState(max_predict_ms=int(predict_ms))
    send_interval = 1.0 / float(send_hz)

    # cached frame-shape (only set after the first decoded frame)
    last_w = last_h = None
    # cached for the ESP overlay between detection frames -- so the boxes don't all vanish
    # on a tick where no fresh detection arrived
    last_boxes = []
    last_engaging = False

    last_send = 0.0
    frames = sends = targets = 0
    last_target_id = None

    while max_frames is None or sends < max_frames:
        # Wait either for a new frame OR up to one send interval (whichever comes first)
        slice_s = max(0.001, send_interval - (time.monotonic() - last_send))
        png, eof = recv.take(timeout=slice_s)
        if eof:
            break

        # 1) DETECTION LEG -- only fires when a fresh frame arrived
        if png is not None:
            frame = None
            dets = []
            target = None
            try:
                frame = P.decode_frame(png)
                if frame is not None:
                    last_h, last_w = frame.shape[:2]
                    dets = detector.detect(frame)[0]
                    if smoother is not None:
                        dets = smoother.update(dets)
                    target = tracker.select(dets, last_w, last_h, fov)
                    # update the high-freq bearing state ONLY from in-cone (engaging) detections;
                    # an approach target is "turn toward this off-cone box" not "shoot here yet"
                    if target is not None and tracker.engaging:
                        dy, dp = bearing_from_bbox(target.cx, target.cy, last_w, last_h, fov)
                        now_ms = time.monotonic() * 1000.0
                        state.on_measurement(dy, dp, target.h, target.w, target.conf, now_ms)
                    elif target is None or not tracker.engaging:
                        # losing the engage target: let TargetState's predict_ms grace decide when
                        # to actually drop. Don't snap-reset here -- a 1-frame detection miss in
                        # the middle of a shot must not zero the bearing.
                        pass
                    last_engaging = bool(target is not None and tracker.engaging)
                    last_boxes = boxes_payload(dets, target, last_engaging)
                if debug:
                    print(f"[recv] frame#{frames} bytes={len(png)} det={len(dets)} "
                          f"target={target is not None} engaging={last_engaging}")
            except Exception as e:
                print(f"[frame {frames}] inference skipped ({type(e).__name__}: {e})")
            if show and frame is not None:
                try:
                    _show_detections(frame, dets, target, None)
                except Exception:
                    pass
            frames += 1

        # 2) CONTROL LEG -- runs at send_hz regardless of frame arrival
        now = time.monotonic()
        if now - last_send < send_interval:
            continue
        last_send = now
        now_ms = now * 1000.0

        sol = None
        approach_target_in_dets = None     # if not engaging, we may still want to pan toward an approach target
        if state.has_target(now_ms) and last_w is not None and last_h is not None:
            dy, dp, h_px, w_px, c = state.current_bearing()
            sol = aim_from_bearing(dy, dp, h_px, k=k, frame_h=last_h, fov_deg=fov)
            # account for the turn the mod is about to make from THIS action: subtract the
            # expected actual movement (mirrored gain + clamp + deadzone) from our held bearing
            state.on_send(sol.d_yaw, sol.d_pitch)
        # ESP overlay & "look to the other side" pan only matter when we have NO engage target
        # AND there's a current TargetTracker approach selection -- handled inside the detection
        # leg above by simply not updating state. If the tracker holds an approach selection on
        # the most recent frame, fall back to the old per-frame approach output:
        if sol is None and tracker._last is not None and not tracker.engaging and last_w is not None:
            # build a one-shot approach solution from the tracker's last detection (no
            # TargetState involvement -- approach is intentionally low-precision)
            t = tracker._last
            from .aim import aim_at as _aim_at_now
            sol_app = _aim_at_now(t, last_w, last_h, fov, k=k)
            s = APPROACH_STEP_DEG
            sol = replace(sol_app, fireable=False,
                          d_yaw=max(-s, min(s, sol_app.d_yaw)),
                          d_pitch=max(-s, min(s, sol_app.d_pitch)))

        action = aim_to_action(sol, n_det=len(last_boxes))
        action["boxes"] = last_boxes
        try:
            P.send_msg(sock, P.encode_action_bin(action))
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise            # surfaces to main()'s reconnect loop
        sends += 1
        targets += 1 if sol is not None else 0
        if debug:
            age = int(state.age_ms(now_ms)) if state.has_target(now_ms) else -1
            print(f"[send#{sends}] d_yaw={action['d_yaw']} d_pitch={action['d_pitch']} "
                  f"range={action['range']} fire={action['fire_ok']} age={age}ms")
        if on_step is not None:
            on_step(sends, len(last_boxes), sol, action)

    recv.close()
    return sends, targets


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
                    help="inference size; default auto = 416 on GPU (the source frame is 427x240 — pushing to "
                         "640 only upsamples height with no extra signal) / 320 on CPU (speed). Pass 640 to "
                         "match the trained size on a beefy GPU, or 256 for the lightest CPU runs.")
    # Frame-to-frame detection smoother — cheap remedy for the weak detector's flicker on far targets.
    # v0.4 async-tracker knobs: control rate is decoupled from detection rate.
    ap.add_argument("--send-hz", type=float, default=20.0,
                    help="rate at which Python emits actions (Hz). Decoupled from detection rate, so "
                         "even at 10 fps detection the mod gets a fresh predicted action every 50 ms. "
                         "Cap is the mod's tick rate (20 Hz); going higher just wastes packets.")
    ap.add_argument("--predict-ms", type=int, default=300,
                    help="how long TargetState holds a target after the last fresh detection (ms). "
                         "Beyond this the target is dropped and the bow stops. Default 300 ms.")
    ap.add_argument("--no-smooth", action="store_true",
                    help="disable the IoU-tracker detection smoother (default ON). The smoother holds a "
                         "missing detection alive for --smooth-miss frames and EMA-smooths the box.")
    ap.add_argument("--smooth-miss", type=int, default=2,
                    help="frames to hold a missing detection before dropping it (default 2 = ~200ms at 10fps)")
    ap.add_argument("--smooth-iou", type=float, default=0.3,
                    help="IoU threshold for considering two boxes the same object across frames (default 0.3)")
    ap.add_argument("--smooth-ema", type=float, default=0.7,
                    help="EMA factor for box-coord smoothing; 1.0 = no smoothing, 0.0 = freeze (default 0.7)")
    ap.add_argument("--smooth-hits", type=int, default=1,
                    help="frames a NEW detection must survive before being surfaced; 1=immediate (default), "
                         "2+ suppresses single-frame false positives at the cost of 1 frame of latency")
    ap.add_argument("--backend", choices=("auto", "ultralytics", "onnxruntime"), default="auto",
                    help="inference backend. auto = .pt -> Ultralytics, .onnx -> onnxruntime (with "
                         "DirectML/CUDA/CPU provider auto-pick) if installed, else Ultralytics. "
                         "Direct ORT is faster on .onnx at small imgsz and unlocks DirectML on Windows.")
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
    imgsz = a.imgsz if a.imgsz is not None else (416 if str(device).startswith("cuda") else 320)
    print(f"[runtime] device={device} imgsz={imgsz} conf={a.conf}")

    from .runtime import make_detector
    detector = make_detector(a.weights, conf=a.conf, device=device, imgsz=imgsz, backend=a.backend)
    smoother = None if a.no_smooth else DetectionSmoother(
        iou_thresh=a.smooth_iou, max_miss=a.smooth_miss, ema=a.smooth_ema, min_hits=a.smooth_hits)
    if smoother is not None:
        print(f"[runtime] smoother on: iou>={a.smooth_iou} max_miss={a.smooth_miss} "
              f"ema={a.smooth_ema} min_hits={a.smooth_hits}")

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
                           on_step=status, debug=a.debug_protocol, show=a.show, smoother=smoother,
                           send_hz=a.send_hz, predict_ms=a.predict_ms)
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
