mc-bow-agent v0.3.0 — yolov8s + DirectML + raw-frame protocol + detection smoother

Trained: 2026-06-30 18:22
Dataset: 4 runs from 2026-06-30, 3050 frames / 13635 boxes (episode-split 3/1).
Model:   YOLOv8s @ imgsz 640, 40 epochs, dynamic-shape ONNX export.

Metrics (val on held-out episode):
  mAP50      0.42  (v2 nano 6921-frame: 0.39)
  mAP50-95   0.19
  Precision  0.58  (v2: ~0.45) — main win: fewer false positives
  Recall     0.43  (v2: ~0.45) — flat, smaller dataset

Mod features bundled in mcbowagent-v0.3.0.jar:
  - raw-BGR frame protocol (magic 'R'); binary action ('A')
  - pipelined socket (mod writer/reader threads; py latest-wins recv)
  - in-game ESP overlay (F9), captured-before-overlay
  - 10Hz default record sample, async PNG writer (no F8 lag)
  - oracle (F10) with ballistic drop solver (Ballistic.java)
  - CRC32 frame dedup, NaN-guarded view writes
  - bow control: 2-consecutive-aligned fire gate, post-fire cooldown

Python runtime:
  - OrtDetector (DirectML/CUDA/CPU auto-pick, YOLOv8-shape probe)
  - DetectionSmoother (IoU tracker, hold-through-miss, EMA box)
  - TargetTracker (crosshair-nearest + FOV cone + switch hysteresis)

Run:
  python -m mc_bow_agent.runtime_loop --weights releases/v0.3.0-yolov8s/mcbow_zombie_v3.onnx --device cuda:0 --imgsz 640
