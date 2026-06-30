"""Synthetic smoke test for dataset.py — PIL only, no ultralytics/torch.

Builds a tiny fake run (2 episodes, PNG frames + matching JSONL with known
bboxes), runs build_dataset, and asserts the YOLO labels, episode split, and
data.yaml. Run from python/:  python -m mc_bow_agent.selftest_dataset
"""
import json
import shutil
import tempfile
from pathlib import Path

from .dataset import build_dataset, bbox_to_yolo


def _png(path, w, h):
    from PIL import Image
    Image.new("RGB", (w, h), (40, 40, 40)).save(path)


def _rec(tick, frame, mobs):
    return {"tick": tick, "frame_path": frame, "player_xyz": [0, 0, 0],
            "player_yaw": 0.0, "player_pitch": 0.0, "health": 20.0, "arrows": 64,
            "bow_charge_ticks": 0, "mobs": mobs,
            "action": {"camera": [0, 0], "forward": 0, "back": 0, "left": 0, "right": 0,
                       "jump": 0, "sprint": 0, "sneak": 0, "use": 0, "hotbar": 1,
                       "target_entity_id": -1},
            "events": {"arrow_released": False, "arrow_hit": False, "kill": False,
                       "damage_taken": False}}


def _mob(type_, bbox, visible=True):
    return {"type": type_, "entity_id": 1, "world_xyz": [0, 0, 0], "velocity": [0, 0, 0],
            "health": 20.0, "rel_yaw": 0.0, "rel_pitch": 0.0, "distance": 10.0,
            "radial_speed": 0.0, "tangential_speed": 0.0, "visible": visible, "screen_bbox": bbox}


def test_bbox_to_yolo():
    cx, cy, w, h = bbox_to_yolo([20, 40, 60, 140], 100, 200)
    assert abs(cx - 0.4) < 1e-6 and abs(cy - 0.45) < 1e-6, (cx, cy)
    assert abs(w - 0.4) < 1e-6 and abs(h - 0.5) < 1e-6, (w, h)
    assert bbox_to_yolo([10, 10, 10, 20], 100, 200) is None       # zero width
    # unsorted corners are normalised, then clamped
    assert bbox_to_yolo([60, 140, 20, 40], 100, 200) is not None
    # partially out-of-bounds -> clip FIRST, keep the visible centre (not the corner)
    cx, cy, w, h = bbox_to_yolo([50, 50, 150, 150], 100, 100)
    assert abs(cx - 0.75) < 1e-6 and abs(w - 0.5) < 1e-6, (cx, w)


def test_build_dataset():
    tmp = Path(tempfile.mkdtemp(prefix="mcbow_ds_"))
    try:
        runs = tmp / "runs"
        for ep in (1, 2):
            run = runs / f"run_2026010{ep}_000000"
            (run / "frames").mkdir(parents=True)
            recs = []
            for t in range(3):
                fp = f"frames/frame_{t:06d}.png"
                _png(run / fp, 64, 64)
                if t == 0:
                    mobs = [_mob("zombie", [10, 10, 30, 50], True)]
                elif t == 1:
                    mobs = [_mob("zombie", [5, 5, 20, 40], True),
                            _mob("creeper", [40, 40, 60, 60], False)]   # occluded -> dropped
                else:
                    mobs = []                                           # background frame
                recs.append(_rec(t, fp, mobs))
            (run / "episode_0001.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

        out = tmp / "dataset"
        stats, leaky = build_dataset([runs], out,
                                     classes=["zombie", "skeleton", "creeper", "spider"],
                                     only_visible=True, val_ratio=0.5, keep_empty=True, seed=0)

        assert not leaky, "2 episodes must split by episode"
        y = (out / "data.yaml").read_text(encoding="utf-8")
        assert "0: zombie" in y and "names:" in y, y

        imgs = list((out / "images" / "train").glob("*.png")) + \
               list((out / "images" / "val").glob("*.png"))
        assert len(imgs) == 6, len(imgs)                              # 2 episodes x 3 frames

        lbls = list((out / "labels" / "train").glob("*.txt")) + \
               list((out / "labels" / "val").glob("*.txt"))
        assert len(lbls) == 4, len(lbls)                             # 2 zombie frames per episode
        for lf in lbls:
            for ln in lf.read_text().strip().splitlines():
                assert ln.split()[0] == "0", ln                       # only zombie (class 0)

        assert stats["background"] == 2, stats                        # the two empty frames
        assert stats["boxes"] == 4, stats
        assert stats["mismatch_warn"] == 0, stats

        # episode-level split: train images and val images come from different runs
        train_runs = {p.name.split("_episode")[0] for p in (out / "images" / "train").glob("*.png")}
        val_runs = {p.name.split("_episode")[0] for p in (out / "images" / "val").glob("*.png")}
        assert train_runs and val_runs and train_runs.isdisjoint(val_runs), (train_runs, val_runs)
        print("dataset stats:", stats)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL {len(tests)} DATASET TESTS PASSED")


if __name__ == "__main__":
    main()
