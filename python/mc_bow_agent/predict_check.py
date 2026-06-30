"""Sanity-check a trained detector on recorded frames: run inference on a few
frames (closest zombies first), save annotated images, and print detected vs
ground-truth counts. CPU by default to avoid CUDA commit-memory pressure.

  python -m mc_bow_agent.predict_check --weights runs/detect/mcbow_zombie_v1/weights/best.pt \
      --run ../runs/run_20260629_051905 --n 8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _max_gt_area(rec):
    best = 0
    for m in rec.get("mobs", []):
        b = m.get("screen_bbox")
        if b and m.get("visible"):
            best = max(best, (b[2] - b[0]) * (b[3] - b[1]))
    return best


def _gt_count(rec):
    return sum(1 for m in rec.get("mobs", []) if m.get("visible") and m.get("screen_bbox"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Visualize detector output on recorded frames.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run", required=True, help="a recording run dir (has episode_*.jsonl + frames/)")
    ap.add_argument("--n", type=int, default=8, help="frames to check (closest + mid distance)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/predict_check")
    a = ap.parse_args(argv)

    run = Path(a.run)
    ep = next(run.glob("episode_*.jsonl"))
    recs = [json.loads(l) for l in open(ep, encoding="utf-8") if l.strip()]
    withz = sorted((r for r in recs if _max_gt_area(r) > 0), key=_max_gt_area, reverse=True)
    if not withz:
        raise SystemExit("no frames with visible zombies found")
    mid = len(withz) // 2
    picks = withz[: a.n // 2] + withz[mid: mid + (a.n - a.n // 2)]

    from ultralytics import YOLO
    model = YOLO(a.weights)
    print(f"{'frame':<22} {'GT':>3} {'DET':>3}  maxGTarea  confidences")
    tot_gt = tot_det = 0
    for r in picks:
        img = str(run / r["frame_path"])
        res = model.predict(img, conf=a.conf, device=a.device, verbose=False,
                            save=True, project=a.out, name="p", exist_ok=True)[0]
        n_gt = _gt_count(r)
        n_det = len(res.boxes)
        tot_gt += n_gt
        tot_det += n_det
        confs = [round(float(c), 2) for c in res.boxes.conf.tolist()]
        print(f"{os.path.basename(img):<22} {n_gt:>3} {n_det:>3}  {_max_gt_area(r):>8}  {confs}")
    print(f"\ntotals: GT={tot_gt}  DET={tot_det}")
    print("annotated frames saved to:", str(Path(a.out) / "p"))


if __name__ == "__main__":
    main()
