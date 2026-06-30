# GPU acceleration via onnxruntime-gpu

The default `onnxruntime-directml` build runs the YOLO ONNX through Microsoft's DirectML —
it cooperates with Minecraft's GPU on Windows but is the slowest GPU path. Swap to
`onnxruntime-gpu` to unlock the CUDA + TensorRT execution providers; `OrtDetector` auto-picks
the fastest available.

| Path | Latency on RTX 4060 @ 640×640 | Notes |
|---|---|---|
| DirectML (default) | ~50 ms | works anywhere, slow |
| CUDA EP | **~8 ms** | one `pip install` (~6× DML) |
| TensorRT EP (FP16) | **~3 ms** | first run compiles + caches an engine under `.trt_cache/` (~18× DML) |

You don't build an `.engine` yourself — onnxruntime's TensorRT EP does it on the first
inference and caches the binary next to the weights. After that every run loads in <5 s.

---

## Install (the version matrix matters)

`pip install onnxruntime-gpu` (no version pin) gets the LATEST, which currently demands
**CUDA 13.x + cuDNN 9.x**. Most users on CUDA 11/12 must pin a compatible version or the
GPU EPs silently fail and the runtime falls back to CPU (~7 fps, painful).

| Your CUDA | Install command | cuDNN |
|---|---|---|
| **12.x** (most current installs) | `pip install "onnxruntime-gpu==1.22.0"` | `pip install nvidia-cudnn-cu12` |
| **11.8** | `pip install "onnxruntime-gpu==1.18.1"` | cuDNN 8.9 for CUDA 11 (NVIDIA dev site) |
| **13.x** (new install) | `pip install onnxruntime-gpu` (latest) | `pip install nvidia-cudnn-cu12` |

Check yours: `nvcc --version`. **Driver version (from `nvidia-smi`) is NOT the toolkit
version** — `nvidia-smi` reports the driver-supported CUDA, not what's actually installed.

For the full CUDA-12 quickstart that worked on the dev box (RTX 4060 Laptop, CUDA 12.6):
```powershell
# C:\ on Windows is usually tiny; redirect pip TEMP to a roomy drive
$env:TMP = "D:\pip-tmp"; $env:TEMP = "D:\pip-tmp"; $env:PIP_CACHE_DIR = "D:\pip-cache"
New-Item -ItemType Directory -Force D:\pip-tmp, D:\pip-cache | Out-Null

pip uninstall -y onnxruntime-directml onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu==1.22.0"
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 `
            nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12

# OPTIONAL: also enable TensorRT EP (best speed). Match CUDA build (10.x for CUDA 12, 11.x for CUDA 13):
pip install "tensorrt-cu12==10.7.0"

python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expect:  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

`OrtDetector._add_nvidia_dll_dirs` automatically adds `site-packages/nvidia/*/bin` and
`site-packages/tensorrt_libs/` to PATH at session creation, so you don't have to edit
your system PATH.

---

## Run

```powershell
python -m mc_bow_agent.runtime_loop --weights runs/.../best.onnx --device cuda:0
```

You'll see the chosen providers printed at startup:
```
[OrtDetector] providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
              imgsz=640  nc=1  conf=0.25
```

First TRT-EP run compiles an engine (2-5 min); subsequent runs load instantly from
`.trt_cache/` next to the weights.

---

## Troubleshooting

- **`providers=['CPUExecutionProvider']`** at startup → GPU EPs failed to load. Common causes:
  - `Require cuDNN 9.* and CUDA 13.*` → installed onnxruntime-gpu is too new; pin per the table.
  - `nvinfer_NN.dll missing` → TensorRT runtime DLLs not on PATH. Either `pip install tensorrt-cu12==10.7.0` (matches ORT 1.22 for CUDA 12) or skip TRT and keep just CUDA EP — still ~6× DML.
  - `cufft64_NN.dll missing` → install the corresponding `nvidia-cufft-cu12` wheel.
- **Build OOMs during first TRT-EP run** → workspace too big. The defaults in `OrtDetector` (`trt_max_workspace_size = 1 << 30` = 1 GB, `trt_builder_optimization_level = 3`) work on 8 GB cards; lower further on smaller GPUs.
- **No GPU at all** → drop `--device cuda:0` and run on CPU; the agent still works, just slower.

Engines are GPU/driver-specific. Rebuild (`rm -rf runs/.../weights/.trt_cache/`) after a
major driver update or moving to a different machine.
