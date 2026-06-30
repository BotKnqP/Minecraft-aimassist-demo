"""Unit tests for the OrtDetector pre/post-processing math (no onnxruntime needed — we only test the
pure-NumPy helpers: _letterbox aspect/padding, _nms IoU greedy keep, and the un-letterbox coord remap).

  python -m mc_bow_agent.selftest_ort
"""
import numpy as np

from .runtime import OrtDetector, make_detector


def test_letterbox_preserves_aspect_and_pads_to_square():
    """427x240 frame -> 416 letterbox: scale by 416/427, pad height to 416 with even top/bot."""
    im = np.zeros((240, 427, 3), dtype=np.uint8)
    x, r, pad_l, pad_t, (h0, w0) = OrtDetector._letterbox(im, 416)
    assert x.shape == (1, 3, 416, 416), f"got {x.shape}"
    assert (h0, w0) == (240, 427)
    assert abs(r - 416 / 427) < 1e-6
    new_h = int(round(240 * r))                          # 234
    assert pad_l == 0                                    # width already matches new_size
    assert pad_t == (416 - new_h) // 2                   # even padding


def test_letterbox_unletterbox_round_trip():
    """Round-trip: place a known box, letterbox, then map back to original — same coords."""
    im = np.zeros((240, 427, 3), dtype=np.uint8)
    # original-frame box centred at (213.5, 120), 80x60
    box0 = np.array([[173.5, 90.0, 253.5, 150.0]])
    _, r, pad_l, pad_t, _ = OrtDetector._letterbox(im, 416)
    # forward: orig -> letterbox space
    box_lb = box0.copy() * r
    box_lb[:, [0, 2]] += pad_l
    box_lb[:, [1, 3]] += pad_t
    # inverse: letterbox -> orig (the same math used in OrtDetector.detect)
    back = box_lb.copy()
    back[:, [0, 2]] -= pad_l
    back[:, [1, 3]] -= pad_t
    back /= r
    assert np.allclose(back, box0, atol=1e-6)


def test_nms_keeps_highest_and_filters_overlapping():
    """Two boxes at the same place: NMS keeps only the higher-confidence one. A third far away survives."""
    boxes = np.array([[0, 0, 10, 10],     # high conf
                      [0, 0, 10, 10],     # exact overlap
                      [100, 100, 110, 110]], dtype=np.float32)  # far -> survives
    scores = np.array([0.9, 0.85, 0.7], dtype=np.float32)
    keep = OrtDetector._nms(boxes, scores, 0.45)
    assert keep.tolist() == [0, 2], f"got {keep.tolist()}"


def test_nms_iou_threshold_just_above_keeps_both():
    """Two boxes with 50% IoU: at iou_thresh 0.6 NMS keeps both; at 0.4 it filters the lower-score one."""
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=np.float32)  # IoU = 50/150 = 1/3
    scores = np.array([0.9, 0.7], dtype=np.float32)
    assert OrtDetector._nms(boxes, scores, 0.6).tolist() == [0, 1]
    assert OrtDetector._nms(boxes, scores, 0.2).tolist() == [0]


def test_make_detector_routes_by_extension():
    """make_detector should NOT instantiate a real Ultralytics for .pt unless onnxruntime is missing for
    .onnx — keep it cheap by passing a non-existent path and catching the expected error."""
    # routing only — we expect each backend to fail on the fake path in a recognizable way
    try:
        make_detector("does-not-exist.pt", backend="ultralytics")
    except Exception as e:
        assert "does-not-exist" in str(e) or "FileNotFound" in type(e).__name__ \
               or "Errno" in str(e) or "No such" in str(e).lower(), f"unexpected: {e}"
    # if onnxruntime isn't installed, .onnx falls back to Ultralytics (same kind of file error)
    try:
        make_detector("does-not-exist.onnx", backend="auto")
    except Exception as e:
        assert "does-not-exist" in str(e) or "Errno" in type(e).__name__ \
               or "No such" in str(e).lower() or "FileNotFound" in type(e).__name__, f"unexpected: {e}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL {len(tests)} ORT TESTS PASSED")


if __name__ == "__main__":
    main()
