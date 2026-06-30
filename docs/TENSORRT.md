# GPU acceleration: ORT-CUDA vs TensorRT

The default `onnxruntime-directml` build runs the YOLO ONNX through Microsoft's DirectML
backend — it cooperates with Minecraft's GPU on Windows but is the SLOWEST GPU path. Two
upgrades are available, both optional, both pick up automatically once installed.

| Path | Latency on RTX 4060 @ 640×640 | Setup difficulty | Why pick it |
|---|---|---|---|
| **A. onnxruntime-gpu (CUDA EP)** | ~25–40 ms | one `pip` | 30–50 % faster than DML, near-zero work |
| **B. TensorRT FP16 engine** | **~8–15 ms** | install TRT SDK + build engine | 3–4× faster than DML, peak inference perf |

You can run either path WITHOUT changing the rest of the pipeline — `make_detector` routes
`.onnx` → ORT and `.engine` → TensorRT automatically.

---

## Path A — onnxruntime-gpu (CUDA / TensorRT EP)

Replaces the DirectML EP with CUDA + (bundled) TensorRT. Same .onnx file, faster runtime.

### ⚠️ Version matrix (read this before installing)

`pip install onnxruntime-gpu` (no version pin) gets the LATEST, which currently demands
**CUDA 13.x + cuDNN 9.x** — most users on CUDA 11/12 will fall straight back to CPU and get
~7 fps instead of 30+. Pick the version that matches your toolkit:

| Your CUDA | Install command | Needs cuDNN |
|---|---|---|
| **12.x** (most current installs) | `pip install onnxruntime-gpu==1.19.2` | cuDNN **8.9** for CUDA 12 |
| **11.8** | `pip install onnxruntime-gpu==1.18.1` | cuDNN **8.9** for CUDA 11 |
| **13.x** (new install) | `pip install onnxruntime-gpu` (latest) | cuDNN **9.x** |

Check yours: `nvcc --version`. **Driver version (from `nvidia-smi`) is NOT the toolkit version** —
nvidia-smi reports the driver-supported CUDA, not what's actually installed.

### Install

```powershell
# remove DirectML build (the two packages can't coexist cleanly)
pip uninstall -y onnxruntime-directml onnxruntime
# install the CUDA build matching your toolkit (CUDA 12.x example)
pip install onnxruntime-gpu==1.19.2
# install cuDNN via pip wheel (avoids the NVIDIA developer-site download dance)
pip install nvidia-cudnn-cu12
# verify
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# OR (if no TRT EP):  ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If you see CUDA (or TRT) in the list, you're set. Run the agent:
```powershell
python -m mc_bow_agent.runtime_loop --weights runs/.../best.onnx --device cuda:0
```
`OrtDetector` will print the providers it actually chose. If you get
`providers=['CPUExecutionProvider']` instead, the GPU EPs failed to load — go back to DML:
```powershell
pip uninstall -y onnxruntime-gpu
pip install onnxruntime-directml
```

### Troubleshooting

- **`Error 126: ...nvinfer_xx.dll missing`** — TensorRT runtime DLLs not on PATH. Either install the
  full TensorRT SDK (Path B below) or stick with the CUDA EP (still ~1.5× DML).
- **`Require cuDNN 9.* and CUDA 13.*`** — you installed too-new onnxruntime-gpu. Downgrade per the
  table above.
- **`Failed to create CUDAExecutionProvider`** — CUDA toolkit + cuDNN aren't both on PATH. If you
  `pip install nvidia-cudnn-cu12`, add `D:\Python\Lib\site-packages\nvidia\cudnn\bin` to PATH.

---

## Path B — TensorRT engine

Loads a prebuilt `.engine` file (CUDA kernels compiled offline for your specific GPU/precision).

### 1. Install TensorRT

NVIDIA TensorRT 8.6+ or 10.x. Two ways:

**Easy path** (TRT 10+, RTX 40-series):
```powershell
pip install tensorrt cuda-python
```
Wheels include the runtime. You can build engines via the Python API (`build_engine.py` below)
without needing the full SDK / `trtexec`.

**Full SDK** (any GPU, also gives you `trtexec` for diagnostics):
1. Go to https://developer.nvidia.com/tensorrt (NVIDIA account required)
2. Download TensorRT 10.x ZIP for Windows + your CUDA version
3. Unzip to e.g. `D:\TensorRT-10.x.y.z\`
4. Add `D:\TensorRT-...\lib` to your `PATH`
5. `pip install D:\TensorRT-...\python\tensorrt-*-cp312-*.whl` (match your Python version)
6. `pip install cuda-python`
7. Verify: `python -c "import tensorrt as trt; print(trt.__version__)"`

### 2. Build the engine from your ONNX

```powershell
cd D:\projects\mc-bow-agent\python
python -m mc_bow_agent.build_engine `
    --onnx runs/detect/mcbow_zombie_v3/weights/best.onnx `
    --output runs/detect/mcbow_zombie_v3/weights/best.engine `
    --fp16
```
This takes 2–10 minutes (TRT tunes hundreds of kernels for your specific GPU). Output is a
`best.engine` of ~30–50 MB.

**Dynamic shape ONNX** (default since v0.4 `train.py`): the builder sets up a dynamic profile
with `--min-imgsz 320 --opt-imgsz 640 --max-imgsz 640`. The engine accepts any input size in
that range, and TRT picks the best kernels for the `opt` size. Pass `--max-imgsz 416` etc. if
you want to lock the engine smaller for more speed.

**Static shape ONNX**: the engine inherits the ONNX's fixed input shape — fastest, but
you can only run at that exact `imgsz`.

### 3. Run

```powershell
python -m mc_bow_agent.runtime_loop `
    --weights runs/detect/mcbow_zombie_v3/weights/best.engine `
    --device cuda:0
```
You'll see `[TrtDetector] loaded engine ...` at startup. The 20 Hz control rate should now be
trivially achievable; if you raise `--send-hz 30` or higher, the mod's tick rate (20 Hz) becomes
the bottleneck, not Python.

### Caveats

- **Engines are GPU/driver-specific.** An engine built on RTX 4060 will work on another 4060
  with a similar driver, but not on different architectures. Rebuild after a major driver update.
- **FP16 precision**: tested OK for YOLOv8 detection; can lower confidence by ~1–2 % vs FP32 in
  rare cases. INT8 would be another 1.5–2× speedup but needs calibration data — not implemented
  here. (`--int8` flag exists but errors out.)
- **TensorRT/cuda-python install is fiddly on Windows.** If the wheel install fails, the SDK
  download path always works.

---

## Choosing

- **Just want a clear win?** Path A. 5 minutes, ~1.5× speedup.
- **Want the peak?** Path B. Half an hour the first time, ~3–4× speedup, lower latency
  variance → smoother control loop.
- **Both fall back gracefully**: if any GPU path fails (OOM, missing provider, bad engine),
  the runtime falls back to CPU automatically; the bot keeps running, just slower.
