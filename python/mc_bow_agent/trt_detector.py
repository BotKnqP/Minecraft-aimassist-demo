"""TensorRT detector — loads a prebuilt .engine and runs YOLO inference on the GPU.

Optional dependency: requires NVIDIA TensorRT + cuda-python installed. The Detector class
is imported lazily by runtime.make_detector when it sees a .engine weights file, so users
without TensorRT can ignore this module entirely and the rest of the codebase works fine.

Build the engine once from an .onnx:
    python -m mc_bow_agent.build_engine \\
        --onnx runs/detect/.../weights/best.onnx \\
        --output runs/detect/.../weights/best.engine \\
        --fp16

Then run:
    python -m mc_bow_agent.runtime_loop --weights runs/.../best.engine --device cuda:0

See docs/TENSORRT.md for install + troubleshooting.
"""
from __future__ import annotations

import os

from .aim import Detection
from .runtime import OrtDetector   # reuse letterbox + NMS (pure NumPy, no ORT/TRT coupling)


class TrtDetector:
    """Drop-in detector replacement for OrtDetector that runs a TensorRT engine instead of
    onnxruntime. Same input/output contract:
      detect(frame_bgr) -> (list[Detection], (h, w))

    Assumes the engine was built from a YOLOv8 single-class export (matches our v3 weights).
    The engine's I/O bindings are introspected at __init__; mismatched shapes raise loudly."""

    def __init__(self, engine_path: str, conf: float = 0.25, imgsz: int = 640,
                 iou: float = 0.45, max_det: int = 100, device_id: int = 0):
        import tensorrt as trt   # noqa: F401 (verify TensorRT is installed)
        try:
            from cuda import cudart
        except ImportError as e:
            raise ImportError(
                "TrtDetector needs cuda-python for GPU memory mgmt (`pip install cuda-python`). "
                "Original: " + str(e)) from e

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"engine file not found: {engine_path}")

        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.imgsz = int(imgsz)
        self._cudart = cudart

        # set the active CUDA device BEFORE loading the engine
        err, = cudart.cudaSetDevice(device_id)
        _cuda_check(err)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("TensorRT failed to deserialise engine; rebuild for this driver/SDK")
        self.context = self.engine.create_execution_context()

        # discover I/O. For YOLOv8 single-class ONNX the engine should expose 1 input (1,3,S,S)
        # and 1 output (1, 4+nc, N).
        self.input_name = None
        self.output_name = None
        self._inp_shape = None
        self._out_shape = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self._inp_shape = shape
            else:
                self.output_name = name
                self._out_shape = shape
        if self.input_name is None or self.output_name is None:
            raise RuntimeError(f"engine I/O introspection failed: in={self.input_name} out={self.output_name}")

        # If the engine was built with a fixed input shape (typical for trtexec without --minShapes),
        # honour it; OrtDetector already does this for ORT and we mirror the behavior.
        if len(self._inp_shape) == 4 and self._inp_shape[2] > 0 and self._inp_shape[3] > 0:
            ih, iw = int(self._inp_shape[2]), int(self._inp_shape[3])
            if ih != iw:
                raise ValueError(f"TrtDetector: rectangular fixed input {iw}x{ih} not supported")
            if ih != self.imgsz:
                print(f"[TrtDetector] engine input is fixed at {ih}x{ih}; using that instead of imgsz={self.imgsz}")
                self.imgsz = ih

        # YOLOv8 layout sanity check: output should be (1, 4+nc, N) with nc==1
        if len(self._out_shape) != 3 or self._out_shape[0] != 1:
            raise ValueError(f"TrtDetector: unsupported output shape {self._out_shape}")
        c, n = int(self._out_shape[1]), int(self._out_shape[2])
        if c >= n:
            raise ValueError(f"TrtDetector: output {self._out_shape} looks like YOLOv5/v7; rebuild from YOLOv8 .onnx")
        nc = c - 4
        if nc != 1:
            raise ValueError(f"TrtDetector: nc={nc}, only single-class YOLOv8 supported here")

        # allocate persistent host + device buffers (avoid per-frame mallocs)
        import numpy as np
        self._np = np
        self._inp_host = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        self._out_host = np.zeros(self._out_shape, dtype=np.float32)
        err, self._inp_dev = cudart.cudaMalloc(self._inp_host.nbytes)
        _cuda_check(err)
        err, self._out_dev = cudart.cudaMalloc(self._out_host.nbytes)
        _cuda_check(err)
        err, self._stream = cudart.cudaStreamCreate()
        _cuda_check(err)
        self.context.set_tensor_address(self.input_name, int(self._inp_dev))
        self.context.set_tensor_address(self.output_name, int(self._out_dev))
        print(f"[TrtDetector] loaded engine {engine_path}  imgsz={self.imgsz}  nc={nc}  conf={self.conf}")

    def __del__(self):
        # best-effort GPU memory cleanup
        try:
            self._cudart.cudaFree(self._inp_dev)
            self._cudart.cudaFree(self._out_dev)
            self._cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass

    def detect(self, frame):
        """frame: HxWx3 BGR ndarray -> (list[Detection], (h, w)). Same contract as OrtDetector."""
        import numpy as np
        if isinstance(frame, str):
            import cv2
            frame = cv2.imread(frame)
        # reuse OrtDetector's pure-NumPy letterbox + NMS — no ORT coupling, just preprocessing math
        x, r, pad_l, pad_t, (h0, w0) = OrtDetector._letterbox(frame, self.imgsz)
        np.copyto(self._inp_host, x)
        cudart = self._cudart
        err, = cudart.cudaMemcpyAsync(self._inp_dev, self._inp_host.ctypes.data, self._inp_host.nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)
        _cuda_check(err)
        ok = self.context.execute_async_v3(self._stream)
        if not ok:
            raise RuntimeError("TrtDetector: execute_async_v3 failed")
        err, = cudart.cudaMemcpyAsync(self._out_host.ctypes.data, self._out_dev, self._out_host.nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)
        _cuda_check(err)
        err, = cudart.cudaStreamSynchronize(self._stream)
        _cuda_check(err)
        pred = self._out_host[0].T   # (N, 4+nc=5)
        if pred.shape[1] < 5:
            return [], (h0, w0)
        scores = pred[:, 4:].max(axis=1)
        pos = (pred[:, 2] > 0) & (pred[:, 3] > 0)
        mask = (scores >= self.conf) & pos
        if not mask.any():
            return [], (h0, w0)
        pred = pred[mask]
        scores = scores[mask]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        keep = OrtDetector._nms(boxes, scores, self.iou)[: self.max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        boxes[:, [0, 2]] -= pad_l
        boxes[:, [1, 3]] -= pad_t
        boxes /= r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)
        dets = [Detection.from_xyxy(float(boxes[i, 0]), float(boxes[i, 1]),
                                    float(boxes[i, 2]), float(boxes[i, 3]), float(scores[i]))
                for i in range(len(boxes))]
        return dets, (h0, w0)


def _cuda_check(err):
    """Raise on a cudart error code."""
    # cudart enums: error 0 is cudaSuccess
    if int(err) != 0:
        raise RuntimeError(f"cudart error code: {int(err)}")
