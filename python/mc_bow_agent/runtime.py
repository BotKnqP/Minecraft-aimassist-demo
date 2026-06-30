"""Runtime decision brain for the scripted v1 bow agent (no RL).

Perception (YOLO on a frame) -> pick nearest zombie -> aim + range + drop -> action.
The frame SOURCE and action SINK are pluggable so the SAME brain runs on:
  - recorded frames (offline demo / verification, below), and
  - the live game via mss capture + a socket bridge to the mod (integration TODO).

  python -m mc_bow_agent.runtime --weights <best.pt|.onnx> --frame <png>
"""
import argparse

from .aim import Detection, solve_from_detections

DEFAULT_K = 244.3    # fit by calibrate.py on the recordings (distance = k / bbox_height)
DEFAULT_FOV = 70.0   # vertical FOV used by the recorder / ProjectionUtil


class Detector:
    """Thin wrapper over an Ultralytics YOLO (.pt or .onnx).

    GPU-hardened: if `device` asks for CUDA but it isn't available, or the GPU run later fails (most
    commonly an OOM when Minecraft + the CUDA context together exhaust the Windows commit limit), the
    detector transparently FALLS BACK TO CPU for the rest of the run instead of crashing the bot."""

    def __init__(self, weights, conf=0.5, device="cpu", imgsz=640):
        from ultralytics import YOLO
        if str(device).startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[detector] CUDA requested but not available -> using CPU")
                    device = "cpu"
            except Exception:
                device = "cpu"
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    @staticmethod
    def _is_oom(e):
        """A memory/CUDA failure we can recover from by dropping to CPU (vs a real bug we must surface)."""
        if e.__class__.__name__ == "OutOfMemoryError":      # torch.cuda.OutOfMemoryError
            return True
        s = str(e).lower()
        return ("out of memory" in s or "cuda" in s or "cudnn" in s or "cublas" in s
                or ("memory" in s and "alloc" in s))

    def _to_cpu(self):
        """Force CPU for ANY weight format. Resetting the predictor makes the next predict() rebuild the
        backend on CPU — a plain model.to('cpu') raises for an ONNX path (self.model is a str) and would
        leave the CUDA-bound onnxruntime session live, so the OOM would just recur."""
        self.device = "cpu"
        try:
            self.model.predictor = None     # next predict() re-runs setup_model on CPU
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def detect(self, frame):
        """frame: path or HxWx3 ndarray -> (list[Detection], (h, w))."""
        try:
            res = self.model.predict(frame, conf=self.conf, device=self.device,
                                     imgsz=self.imgsz, verbose=False)[0]
        except Exception as e:
            if self.device == "cpu" or not self._is_oom(e):
                raise                       # already CPU, or a real (non-memory) bug -> surface it
            # GPU out of memory (commit/VRAM contention with Minecraft): drop to CPU for the rest of the run
            print(f"[detector] CUDA OOM ({type(e).__name__}: {e}); switching to CPU for the rest of the run")
            self._to_cpu()
            res = self.model.predict(frame, conf=self.conf, device=self.device,
                                     imgsz=self.imgsz, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
            dets.append(Detection.from_xyxy(x0, y0, x1, y1, float(b.conf[0])))
        return dets, res.orig_shape  # orig_shape = (h, w)


def decide(detector, frame, k=DEFAULT_K, fov=DEFAULT_FOV, conf_thresh=0.5):
    """One perception->decision step. Returns (AimSolution|None, n_detections, (w,h))."""
    dets, (h, w) = detector.detect(frame)
    sol = solve_from_detections(dets, w, h, fov, k=k, conf_thresh=conf_thresh)
    return sol, len(dets), (w, h)


# --- integration layer (needs the live game) -------------------------------
# Frame SOURCE options:
#   * mss.grab() of the Minecraft window (Python-side capture), or
#   * the mod streams its 128x128 framebuffer over a localhost socket.
# Action SINK: send {d_yaw, d_pitch, fire} to the mod over the socket; the mod's
#   BowMacroController applies the per-tick clamped turn (<=10 deg/tick) and the
#   use-hold/release. The mod knows the live pitch, so it can refine aim.d_pitch's
#   flat-ground drop assumption with the true target elevation. TODO: implement
#   run_live(detector, source, sink) once the mod's socket endpoint exists.


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the v1 aim brain on a frame.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frame", required=True, help="a recorded frame PNG")
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--fov", type=float, default=DEFAULT_FOV)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    det = Detector(a.weights, conf=a.conf, device=a.device)
    sol, ndet, (w, h) = decide(det, a.frame, k=a.k, fov=a.fov, conf_thresh=a.conf)
    print(f"frame {w}x{h}  detections={ndet}")
    if sol is None:
        print("no target (no zombie above conf)")
    else:
        print(f"TARGET nearest zombie: range={sol.range_blocks:.1f} blocks  "
              f"turn d_yaw={sol.d_yaw:+.1f} d_pitch={sol.d_pitch:+.1f} "
              f"(aimed {sol.drop_deg:.1f} up for drop)  fireable={sol.fireable}")


if __name__ == "__main__":
    main()
