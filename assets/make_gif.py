"""Make the README demo GIF from the recorded mp4 (no ffmpeg needed: OpenCV + Pillow).

  python assets/make_gif.py [--src demo_src.mp4] [--dst demo.gif]
                            [--width 480] [--fps 8] [--colors 128]
                            [--start 0] [--end 0]   # 0 end = whole clip
"""
import argparse
import os

import cv2
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src", default=os.path.join(here, "demo_src.mp4"))
    ap.add_argument("--dst", default=os.path.join(here, "demo.gif"))
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / a.fps))
    start_f = int(a.start * src_fps)
    end_f = int(a.end * src_fps) if a.end > 0 else 10 ** 9

    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start_f <= i < end_f and (i - start_f) % step == 0:
            h, w = frame.shape[:2]
            nh = int(round(h * a.width / w))
            small = cv2.resize(frame, (a.width, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=a.colors))
        i += 1
    cap.release()

    if not frames:
        raise SystemExit("no frames extracted")
    frames[0].save(a.dst, save_all=True, append_images=frames[1:],
                   duration=int(1000 / a.fps), loop=0, optimize=True, disposal=2)
    print(f"frames={len(frames)}  {a.width}px  {a.fps}fps  {a.colors}colors  "
          f"size={os.path.getsize(a.dst) / 1e6:.1f}MB  -> {a.dst}")


if __name__ == "__main__":
    main()
