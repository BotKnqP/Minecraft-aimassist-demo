"""Runtime decision brain for the scripted v1 bow agent (no RL).

Perception (YOLO on a frame) -> pick nearest zombie -> aim + range + drop -> action.
The frame SOURCE and action SINK are pluggable so the SAME brain runs on:
  - recorded frames (offline demo / verification, below), and
  - the live game via the mod's socket bridge to a Python loop.

Two backends ship: `Detector` wraps Ultralytics (works for .pt and .onnx, used by default for .pt) and
`OrtDetector` runs onnxruntime directly (faster path for .onnx, with DirectML/CUDA provider auto-pick).
Use `make_detector(weights, ...)` to get the right one for a given weights file.

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


class OrtDetector:
    """Direct onnxruntime inference for .onnx weights — skips Ultralytics' .predict() wrapper for ~1.5-2x
    less per-call overhead at small imgsz. Provider auto-selection prefers DirectML on Windows (cooperates
    with the desktop GPU Minecraft already owns, unlike CUDA which fights for VRAM commit), then CUDA, then
    CPU; whichever is available + requested falls in. Falls back to CPU silently if the GPU provider isn't
    loaded. The Ultralytics .pt path is untouched."""

    def __init__(self, weights, conf=0.25, device="cpu", imgsz=640, iou=0.45, max_det=100):
        import onnxruntime as ort
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self.max_det = int(max_det)
        available = set(ort.get_available_providers())
        # Honor the user's stated device: when --device cuda* is requested, prefer CUDA over DirectML so
        # multi-GPU CUDA selection (`cuda:1`, etc.) and explicit "I freed VRAM headroom" intent both work.
        # The DirectML preference applies only when device is left at the default (so any GPU is fine).
        want = []
        dev = str(device)
        if dev.startswith("cuda"):
            if "CUDAExecutionProvider" in available:
                want.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available:
                want.append("DmlExecutionProvider")
        want.append("CPUExecutionProvider")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(weights, sess_options=opts, providers=want)
        self.input_name = self.session.get_inputs()[0].name
        ishape = self.session.get_inputs()[0].shape
        # Honor a fixed input H/W if the model has one. Require it to be SQUARE — the letterbox path packs
        # one scalar imgsz; a rectangular fixed input needs per-axis handling and is rare enough to refuse
        # explicitly (clearer than a downstream shape mismatch at the first detect()).
        try:
            ih, iw = int(ishape[2]), int(ishape[3])
        except (TypeError, ValueError, IndexError):
            ih = iw = -1
        if ih > 0 and iw > 0:
            if ih != iw:
                raise ValueError(f"OrtDetector: rectangular fixed input {iw}x{ih} not supported; "
                                 f"export the ONNX at a square size (e.g. yolo export imgsz=640).")
            if ih != self.imgsz:
                print(f"[OrtDetector] model input is fixed at {ih}x{ih}; using that instead of imgsz={self.imgsz}")
                self.imgsz = ih
        # Probe the OUTPUT shape with a dummy zeros frame so we know:
        #   - layout is YOLOv8 (1, 4+nc, N), not YOLOv5/v7 (1, N, 4+1+nc) — refuse the latter loudly
        #   - the model is single-class (nc == 1) — our NMS is class-agnostic; multi-class would cross-suppress
        self._nc = 1
        try:
            import numpy as _np
            probe = _np.zeros((1, 3, self.imgsz, self.imgsz), dtype=_np.float32)
            probe_out = self.session.run(None, {self.input_name: probe})[0]
        except Exception as e:
            raise RuntimeError(f"OrtDetector: dummy inference failed at init ({e}); model may be invalid") from e
        if probe_out.ndim != 3 or probe_out.shape[0] != 1:
            raise ValueError(f"OrtDetector: unsupported output shape {probe_out.shape}; expected (1, 4+nc, N) "
                             f"for YOLOv8.")
        c, n = probe_out.shape[1], probe_out.shape[2]
        # YOLOv8: dim 1 is 4+nc (small), dim 2 is N anchors (large). YOLOv5/v7 swap: dim 1 is N, dim 2 is 4+1+nc.
        if c >= n:
            raise ValueError(f"OrtDetector: output {probe_out.shape} looks like YOLOv5/v7 (or transposed); "
                             f"OrtDetector currently supports YOLOv8 ONNX only — re-export with YOLOv8 or "
                             f"use --backend ultralytics.")
        nc = c - 4
        if nc < 1:
            raise ValueError(f"OrtDetector: output channel count {c} < 5; model has no class scores.")
        if nc > 1:
            raise ValueError(f"OrtDetector: multi-class model (nc={nc}) is not supported by the built-in NMS "
                             f"(it is class-agnostic and would cross-suppress). Use --backend ultralytics, "
                             f"or wait for per-class NMS support.")
        self._nc = nc
        self.providers = self.session.get_providers()
        print(f"[OrtDetector] providers={self.providers}  imgsz={self.imgsz}  nc={nc}  conf={self.conf}")

    @staticmethod
    def _letterbox(im, new_size):
        """Resize-with-pad: preserve aspect to fit a new_size x new_size square. Returns the float CHW
        tensor in [0,1] (RGB), plus the scale `r` and (pad_left, pad_top) needed to un-letterbox boxes."""
        import cv2
        import numpy as np
        h0, w0 = im.shape[:2]
        r = min(new_size / h0, new_size / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        if (new_w, new_h) != (w0, h0):
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = new_size - new_w
        pad_h = new_size - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        if pad_top or pad_bot or pad_left or pad_right:
            im = cv2.copyMakeBorder(im, pad_top, pad_bot, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0   # BGR -> RGB, HWC -> CHW
        return x[np.newaxis], r, pad_left, pad_top, (h0, w0)

    @staticmethod
    def _nms(boxes, scores, iou_thresh):
        """Greedy NMS — single class, NumPy. Returns indices in score-descending order, IoU-filtered."""
        import numpy as np
        if len(boxes) == 0:
            return np.empty(0, dtype=np.int64)
        x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (area[i] + area[rest] - inter + 1e-9)
            order = rest[iou < iou_thresh]
        return np.asarray(keep, dtype=np.int64)

    def detect(self, frame):
        """frame: path or HxWx3 BGR ndarray -> (list[Detection], (h, w))."""
        import numpy as np
        if isinstance(frame, str):
            import cv2
            frame = cv2.imread(frame)
        x, r, pad_l, pad_t, (h0, w0) = self._letterbox(frame, self.imgsz)
        out = self.session.run(None, {self.input_name: x})[0]
        # YOLOv8 ONNX layout is asserted (1, 4+nc, N) at __init__; refuse anything else.
        pred = out[0].T
        if pred.shape[1] < 5:
            return [], (h0, w0)
        cls_scores = pred[:, 4:]
        scores = cls_scores.max(axis=1)
        # confidence threshold + drop degenerate (zero/negative-area) boxes BEFORE NMS so they can't survive
        # as phantom targets (which would feed the approach pan with nothing to actually shoot at)
        pos = (pred[:, 2] > 0) & (pred[:, 3] > 0)
        mask = (scores >= self.conf) & pos
        if not mask.any():
            return [], (h0, w0)
        pred = pred[mask]
        scores = scores[mask]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        keep = self._nms(boxes, scores, self.iou)[: self.max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        # un-letterbox to original frame coords
        boxes[:, [0, 2]] -= pad_l
        boxes[:, [1, 3]] -= pad_t
        boxes /= r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)
        dets = [Detection.from_xyxy(float(boxes[i, 0]), float(boxes[i, 1]),
                                    float(boxes[i, 2]), float(boxes[i, 3]), float(scores[i]))
                for i in range(len(boxes))]
        return dets, (h0, w0)


def make_detector(weights, conf=0.5, device="cpu", imgsz=640, backend="auto"):
    """Pick the backend by file extension + availability:
      .engine -> TrtDetector (TensorRT FP16/INT8; needs `pip install tensorrt cuda-python` plus a CUDA
                 install + a prebuilt engine — see docs/TENSORRT.md and mc_bow_agent.build_engine)
      .onnx   -> OrtDetector (onnxruntime DirectML/CUDA/CPU auto-pick) unless --backend ultralytics
      .pt     -> Detector (Ultralytics; handles the PyTorch graph + OOM fallback)
    """
    w = str(weights).lower()
    is_engine = w.endswith(".engine") or w.endswith(".plan") or w.endswith(".trt")
    is_onnx = w.endswith(".onnx")
    if is_engine:
        device_id = 0
        if isinstance(device, str) and device.startswith("cuda:"):
            try:
                device_id = int(device.split(":", 1)[1])
            except ValueError:
                device_id = 0
        try:
            from .trt_detector import TrtDetector
            return TrtDetector(weights, conf=conf, imgsz=imgsz, device_id=device_id)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(f"TensorRT path requested ({weights}) but TensorRT / cuda-python is not "
                               f"installed: {e}. See docs/TENSORRT.md.") from e
    if backend == "ultralytics" or not is_onnx:
        return Detector(weights, conf=conf, device=device, imgsz=imgsz)
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        if backend == "onnxruntime":
            raise RuntimeError("--backend onnxruntime requested but onnxruntime is not installed "
                               "(pip install onnxruntime  or  onnxruntime-directml)")
        print("[detector] onnxruntime not installed; falling back to Ultralytics for the .onnx weights")
        return Detector(weights, conf=conf, device=device, imgsz=imgsz)
    return OrtDetector(weights, conf=conf, device=device, imgsz=imgsz)


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
