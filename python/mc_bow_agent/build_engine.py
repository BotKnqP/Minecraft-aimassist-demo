"""Build a TensorRT .engine from a YOLOv8 ONNX export, via the TensorRT Python builder API.
No trtexec required — works with the `tensorrt` pip wheel alone (plus a real CUDA install).

  python -m mc_bow_agent.build_engine \\
      --onnx runs/detect/mcbow_zombie_v3/weights/best.onnx \\
      --output runs/detect/mcbow_zombie_v3/weights/best.engine \\
      --fp16

For dynamic-shape ONNX (default since v0.4 train.py export), pass --min/--opt/--max imgsz to
set the dynamic profile. Otherwise the static input shape is reused.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build TensorRT engine from YOLOv8 ONNX")
    ap.add_argument("--onnx", required=True, help="path to the YOLOv8 ONNX (best.onnx)")
    ap.add_argument("--output", required=True, help="output .engine path")
    ap.add_argument("--fp16", action="store_true", help="enable FP16 precision (recommended on Ampere/Ada)")
    ap.add_argument("--int8", action="store_true", help="enable INT8 (needs calibration data; not implemented yet)")
    ap.add_argument("--workspace-gb", type=float, default=2.0,
                    help="max workspace size for TRT optimizer (default 2 GB)")
    ap.add_argument("--min-imgsz", type=int, default=320, help="dynamic-profile min input size (square)")
    ap.add_argument("--opt-imgsz", type=int, default=640, help="dynamic-profile optimal input size")
    ap.add_argument("--max-imgsz", type=int, default=640, help="dynamic-profile max input size")
    a = ap.parse_args(argv)

    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: TensorRT not installed. See docs/TENSORRT.md.", file=sys.stderr)
        return 1

    if not os.path.isfile(a.onnx):
        print(f"ERROR: ONNX not found: {a.onnx}", file=sys.stderr)
        return 1

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(a.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            return 1

    config = builder.create_builder_config()
    # workspace in bytes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    if a.fp16:
        if not builder.platform_has_fast_fp16:
            print("WARN: this GPU does not advertise fast FP16; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    if a.int8:
        print("ERROR: --int8 requested but calibration not implemented; aborting.", file=sys.stderr)
        return 1

    # configure dynamic profile if the network's input is dynamic
    inp = network.get_input(0)
    if inp is None:
        print("ERROR: ONNX has no inputs?", file=sys.stderr)
        return 1
    print(f"input '{inp.name}' shape={tuple(inp.shape)} dtype={inp.dtype}")
    if -1 in tuple(inp.shape):   # dynamic
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name,
                          min=(1, 3, a.min_imgsz, a.min_imgsz),
                          opt=(1, 3, a.opt_imgsz, a.opt_imgsz),
                          max=(1, 3, a.max_imgsz, a.max_imgsz))
        config.add_optimization_profile(profile)
        print(f"dynamic profile: min={a.min_imgsz} opt={a.opt_imgsz} max={a.max_imgsz}")

    print(f"building engine (this can take a few minutes)...  fp16={a.fp16}")
    serialised = builder.build_serialized_network(network, config)
    if serialised is None:
        print("ERROR: TensorRT build failed (see WARN/ERROR above)", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "wb") as f:
        f.write(serialised)
    size_mb = os.path.getsize(a.output) / 1e6
    print(f"wrote {a.output} ({size_mb:.1f} MB)")
    print(f"\nRun:  python -m mc_bow_agent.runtime_loop --weights {a.output} --device cuda:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
