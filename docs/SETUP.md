# Pipeline setup

Tested on Windows 11 with an RTX 4070 Laptop GPU (8 GB). Linux works the same way; swap `Scripts` for `bin`.

## 1. Requirements

| Tool | Version | Notes |
|---|---|---|
| NVIDIA driver | CUDA 11.8+ capable | `nvidia-smi` should work |
| Python | 3.10 (via [uv](https://docs.astral.sh/uv/)) | nerfstudio 1.1.x does not support 3.12 yet |
| FFmpeg | any recent | must be on `PATH` |
| COLMAP | 4.2+ (has the global mapper) | Windows: download `colmap-x64-windows-cuda.zip` from the [releases page](https://github.com/colmap/colmap/releases), unzip it, and add the folder (the one containing `COLMAP.bat`) to `PATH` |
| Node.js | 20+ | for the web viewer |

## 2. Training environment

```bash
uv venv .venv-ns --python 3.10
uv pip install --python .venv-ns/Scripts/python.exe torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv-ns/Scripts/python.exe nerfstudio==1.1.5 "numpy<2"
uv pip install --python .venv-ns/Scripts/python.exe --reinstall --no-deps "https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt21cu118-cp310-cp310-win_amd64.whl"
uv pip install --python .venv-ns/Scripts/python.exe "setuptools<70"
```

- The prebuilt gsplat wheel ships the compiled CUDA kernels, so you don't need a CUDA toolkit or a compiler. The plain PyPI `gsplat` would try to JIT-compile them and fail with `'NoneType' object has no attribute 'CameraModelType'`. (On Linux, use the matching `linux_x86_64` wheel.)
- torch 2.1 still imports `pkg_resources`, which setuptools 70+ removed.
- On a flaky connection, set `UV_HTTP_TIMEOUT=900` first: the torch wheel is 2.5 GB.

Check it:

```bash
.venv-ns/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"   # True
colmap -h
```

## 3. Run

**Easiest:** double-click `start.bat`. It starts the job server and the web app, then opens **Wayfarer Studio** in your browser. Drop the video, enter the ceiling height, press *Build twin*, and open the twin when it finishes.

**Command line:** activate the venv so `ns-*` and the pipeline share it, then:

```bash
.venv-ns/Scripts/activate
python pipeline/wayfarer.py capture.mp4 --name home --ceiling 2.75
```

Typical timings on an RTX 4070 Laptop GPU for a 2-bedroom flat (~350 frames):

| Step | Time |
|---|---|
| frames | < 1 min |
| sfm (COLMAP, sequential matching) | 5–10 min |
| train (splatfacto, 30k iterations) | 20–30 min |
| export + navgrid | < 1 min |

Use `--quality fast` (7k iterations, about 5 min) for a quick check before the full run.

## Troubleshooting

| Symptom | Fix |
|---|---|
| COLMAP registers only part of the frames | Re-shoot with more doorway overlap, or try `--matching exhaustive` |
| CUDA out of memory while training | `--max-side 1280`, or keep `--quality default` instead of `high` |
| Model is tilted or the floor is wrong | The splat's rough up axis differs: run `navgrid.py` by hand with `--up 0,1,0` |
| Scale looks off | Compare the `[ceiling]` and `[camera]` estimates the pipeline prints; pass `--scale` to override |
