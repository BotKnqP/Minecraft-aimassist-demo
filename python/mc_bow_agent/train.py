"""Train Ultralytics YOLO on a recorder-derived dataset and export ONNX.

RTX 4060 (8 GB) defaults: yolov8n, imgsz 640, batch 16 — a single-class zombie
detector converges in a few minutes. Export ONNX for later in-process inference
inside the Fabric mod (ONNX Runtime / DJL).

Examples
  # sanity-check the install + dataset without training:
  python -m mc_bow_agent.train --data <out>/data.yaml --check
  # train + export:
  python -m mc_bow_agent.train --data <out>/data.yaml --epochs 100 --device 0
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train YOLO on the recorder dataset.")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="yolov8n.pt / yolov8s.pt (yolo11* needs ultralytics>=8.3)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' = RTX 4060, or 'cpu'")
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers; 0 is safest on Windows (>0 reloads CUDA DLLs "
                         "per worker subprocess and can hit WinError 1455 paging-file errors)")
    ap.add_argument("--cache", default="",
                    help="'disk' caches decoded images to .npy (much faster epochs at workers=0, "
                         "no subprocess commit cost); 'ram' needs ~all images in RAM; '' = off")
    ap.add_argument("--name", default="mcbow_yolo")
    ap.add_argument("--export", default="onnx", choices=["onnx", "none"])
    ap.add_argument("--check", action="store_true", help="verify install + data.yaml, then exit")
    a = ap.parse_args(argv)

    data = Path(a.data)
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}")

    if a.check:
        import importlib
        importlib.import_module("ultralytics")
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "cpu"
        print(f"torch {torch.__version__}  cuda_available={cuda}  device={name}")
        print(f"data.yaml OK: {data.resolve()}")
        return

    if "yolo11" in a.model or "yolov11" in a.model:
        import ultralytics
        raise SystemExit(f"{a.model} requires ultralytics>=8.3 (have {ultralytics.__version__}); "
                         "use a yolov8 model (e.g. yolov8n.pt) or upgrade ultralytics.")

    import torch
    if a.device != "cpu" and not torch.cuda.is_available():
        print("WARNING: CUDA not available -> training on CPU (much slower). Pass --device cpu to silence.")

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, workers=a.workers, cache=(a.cache or False), name=a.name)
    metrics = model.val()
    print("val:", metrics)
    if a.export != "none":
        # dynamic=True so the runtime OrtDetector can pick the inference imgsz (e.g. 416 for speed,
        # 640 for accuracy) without re-exporting. Fixed-shape was a footgun upstream — locked you into
        # whatever imgsz you happened to pass to train.
        try:
            out = model.export(format=a.export, imgsz=a.imgsz, opset=12, simplify=True, dynamic=True)
        except Exception as e:
            print(f"export with simplify failed ({e}); retrying without simplify...")
            try:
                out = model.export(format=a.export, imgsz=a.imgsz, opset=12, simplify=False, dynamic=True)
            except Exception as e2:
                raise SystemExit(f"ONNX export failed: {e2}\n"
                                 "Install export deps:  pip install onnx onnxslim onnxruntime")
        print("exported:", out)


if __name__ == "__main__":
    main()
