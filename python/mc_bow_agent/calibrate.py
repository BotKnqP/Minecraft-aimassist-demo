"""Fit the bbox-height -> distance model from recordings (data-driven ranging).

For a single mob class of fixed real height, distance * bbox_height = focal * H = k
is (ideally) constant. We have ground-truth distance AND screen bbox for every mob
in the recordings, so we fit k = median(distance * bbox_height) (robust to outliers)
and report the fit error. The runtime then uses distance = k / bbox_height.

  python -m mc_bow_agent.calibrate <runs_dir> [--class zombie] [--min-box 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def collect_pairs(runs, mob="zombie", min_box=3):
    """[(bbox_height_px, true_distance)] for visible mobs of the given class."""
    pairs = []
    for jp in sorted(Path(runs).rglob("episode_*.jsonl")):
        with open(jp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for m in rec.get("mobs", []):
                    if m.get("type") != mob or not m.get("visible"):
                        continue
                    b = m.get("screen_bbox")
                    d = m.get("distance")
                    if not b or d is None:
                        continue
                    h = b[3] - b[1]
                    if h >= min_box and d > 0:
                        pairs.append((h, d))
    return pairs


def fit_k(pairs):
    """k = median(distance * height). Returns (k, n, median_abs_pct_err)."""
    if not pairs:
        raise SystemExit("no (bbox, distance) pairs found")
    products = [d * h for (h, d) in pairs]
    k = statistics.median(products)
    errs = [abs((k / h) - d) / d for (h, d) in pairs]
    return k, len(pairs), statistics.median(errs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit bbox-height -> distance scale k.")
    ap.add_argument("runs", help="recordings dir (recurses episode_*.jsonl)")
    ap.add_argument("--class", dest="cls", default="zombie")
    ap.add_argument("--min-box", type=int, default=3, help="ignore boxes shorter than this (px)")
    a = ap.parse_args(argv)

    pairs = collect_pairs(a.runs, a.cls, a.min_box)
    k, n, med_err = fit_k(pairs)
    # a few spot checks across the range
    by_h = sorted(pairs)
    print(f"fitted k = {k:.1f}  (distance = k / bbox_height)   from n={n} {a.cls} boxes")
    print(f"median abs error = {med_err*100:.1f}%")
    print("spot checks (bbox_h px -> pred dist vs true dist):")
    for idx in (0, len(by_h) // 4, len(by_h) // 2, 3 * len(by_h) // 4, len(by_h) - 1):
        h, d = by_h[idx]
        print(f"  h={h:>4}px  pred={k/h:6.1f}  true={d:6.1f}")
    return k


if __name__ == "__main__":
    main()
