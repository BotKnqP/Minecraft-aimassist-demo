"""Convert Fabric-recorder episodes -> a YOLO detection dataset (+ data.yaml).

Each TickRecord (data_schema.py) carries per-mob screen_bbox (pixel [x0,y0,x1,y1])
+ type. This turns those into YOLO label files and lays out the images/labels/
{train,val} tree Ultralytics expects.

Two correctness rules baked in:
  * split by EPISODE, never by frame (consecutive frames are near-identical, so a
    random frame split leaks and inflates val metrics).
  * the saved frame's pixel size MUST equal the resolution the mod projected
    bboxes in. We read each image's real W,H and warn loudly if a bbox lies
    outside it (a resolution mismatch = useless labels).

CLI:  python -m mc_bow_agent.dataset <runs_dir> -o <out_dataset> [--classes zombie]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
from pathlib import Path

DEFAULT_CLASSES = ["zombie"]                                  # v1 is zombie-only
ALL_V0_CLASSES = ["zombie", "skeleton", "creeper", "spider"]  # reference: the mod's V0 mob types


def _img_size(path: Path):
    """(width, height) of a PNG/JPEG. Pillow if available, else a PNG header read."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (w, h)
    except ImportError:
        with open(path, "rb") as f:
            head = f.read(24)
        if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        raise ValueError(f"cannot read image size (no Pillow, not a PNG): {path}")


def bbox_to_yolo(bbox, W, H):
    """[x0,y0,x1,y1] px -> (cx,cy,w,h) normalized to [0,1]; None if degenerate.

    Corners are CLIPPED to the image FIRST, then normalized. Clamping the centre
    and size independently (the naive way) mis-places a partially out-of-bounds
    box at the corner instead of keeping its visible centre.
    """
    x0, y0, x1, y1 = bbox
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = min(max(x0, 0.0), W)
    x1 = min(max(x1, 0.0), W)
    y0 = min(max(y0, 0.0), H)
    y1 = min(max(y1, 0.0), H)
    w = (x1 - x0) / W
    h = (y1 - y0) / H
    if w <= 0.0 or h <= 0.0:
        return None
    cx = ((x0 + x1) / 2.0) / W
    cy = ((y0 + y1) / 2.0) / H
    return cx, cy, w, h


def find_episodes(inputs):
    """[(episode_key, jsonl_path)] for every episode_*.jsonl under the inputs."""
    eps = []
    for root in inputs:
        for jp in sorted(Path(root).rglob("episode_*.jsonl")):
            eps.append((f"{jp.parent.name}/{jp.stem}", jp))
    return eps


def _write_yaml(out: Path, classes):
    lines = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    for i, c in enumerate(classes):
        lines.append(f"  {i}: {c}")
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(inputs, out_dir, classes=None, only_visible=True,
                  val_ratio=0.2, keep_empty=True, seed=0, min_box_px=2):
    classes = classes or DEFAULT_CLASSES
    out = Path(out_dir)
    cls_index = {c: i for i, c in enumerate(classes)}
    episodes = find_episodes(inputs)
    if not episodes:
        raise SystemExit(f"No episode_*.jsonl found under: {inputs}")

    # split by EPISODE to avoid train/val leakage
    keys = [k for k, _ in episodes]
    random.Random(seed).shuffle(keys)
    by_frame_split = len(set(keys)) < 2
    val_keys = set()
    if not by_frame_split:
        n_val = max(1, int(round(len(set(keys)) * val_ratio)))
        val_keys = set(keys[:n_val])

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "labels": 0, "boxes": 0, "background": 0,
             "skipped_no_image": 0, "skipped_class": 0, "mismatch_warn": 0}
    frame_idx = 0

    for ep_idx, (key, jp) in enumerate(episodes):
        run = jp.parent
        with open(jp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                framerel = rec.get("frame_path")
                if not framerel:
                    continue
                img = run / framerel
                if not img.exists():
                    stats["skipped_no_image"] += 1
                    continue
                try:
                    W, H = _img_size(img)
                except Exception:
                    stats["skipped_no_image"] += 1
                    continue

                is_val = (frame_idx % 5 == 0) if by_frame_split else (key in val_keys)
                split = "val" if is_val else "train"
                frame_idx += 1

                lines_out = []
                for mob in rec.get("mobs", []):
                    bbox = mob.get("screen_bbox")
                    if bbox is None:
                        continue
                    if only_visible and not mob.get("visible", False):
                        continue
                    t = mob.get("type")
                    if t not in cls_index:
                        stats["skipped_class"] += 1
                        continue
                    if bbox[2] > W + 2 or bbox[3] > H + 2 or bbox[0] < -2 or bbox[1] < -2:
                        stats["mismatch_warn"] += 1
                        continue   # gross resolution mismatch -> skip, don't emit a corrupt label
                    if (bbox[2] - bbox[0]) < min_box_px or (bbox[3] - bbox[1]) < min_box_px:
                        continue
                    yb = bbox_to_yolo(bbox, W, H)
                    if yb is None:
                        continue
                    lines_out.append(f"{cls_index[t]} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")

                if not lines_out and not keep_empty:
                    continue

                stem = f"e{ep_idx:03d}_{key.replace('/', '_')}_{int(rec.get('tick', frame_idx)):06d}"
                shutil.copyfile(img, out / "images" / split / (stem + img.suffix.lower()))
                stats["images"] += 1
                if lines_out:
                    (out / "labels" / split / (stem + ".txt")).write_text(
                        "\n".join(lines_out) + "\n", encoding="utf-8")
                    stats["labels"] += 1
                    stats["boxes"] += len(lines_out)
                else:
                    stats["background"] += 1

    if stats["images"] == 0:
        raise SystemExit("ERROR: no valid images after filtering — check --classes, "
                         f"--include-occluded, frame paths, and frame resolution. stats={stats}")
    _write_yaml(out, classes)
    return stats, by_frame_split


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a YOLO dataset from recorder episodes.")
    ap.add_argument("inputs", nargs="+", help="run dir(s) containing episode_*.jsonl (recurses)")
    ap.add_argument("-o", "--out", required=True, help="output dataset dir")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="class names -> ids in order (default: zombie; e.g. --classes zombie skeleton)")
    ap.add_argument("--include-occluded", action="store_true", help="keep boxes with visible=false")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--no-empty", action="store_true", help="drop background frames (no boxes)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    stats, leaky = build_dataset(a.inputs, a.out, a.classes, not a.include_occluded,
                                 a.val_ratio, not a.no_empty, a.seed)
    print("dataset:", stats)
    if stats["images"] and stats["background"] / stats["images"] > 0.7:
        print(f"WARNING: {stats['background']}/{stats['images']} frames are background (no boxes) — "
              "consider --no-empty, or record more frames with on-screen mobs.")
    if leaky:
        print("WARNING: <2 episodes -> frame-level split (train/val LEAKAGE). Record more episodes.")
    if stats["mismatch_warn"]:
        print(f"WARNING: {stats['mismatch_warn']} boxes exceed image bounds -> frame/bbox RESOLUTION "
              "MISMATCH. FrameCapture must save at the SAME resolution ProjectionUtil used.")
    if stats["skipped_no_image"]:
        print(f"NOTE: {stats['skipped_no_image']} records skipped (frame missing) — "
              "expected until FrameCapture writes real images.")
    print(f"data.yaml -> {Path(a.out) / 'data.yaml'}")


if __name__ == "__main__":
    main()
