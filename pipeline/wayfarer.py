"""Wayfarer pipeline: phone video(s) / photos -> web-ready navigable digital twin.

  python pipeline/wayfarer.py capture.mp4 --name home --ceiling 2.75

Steps (each is skipped when its output already exists; --force STEP re-runs from STEP):
  frames   ffmpeg -> keep the sharpest frame per window (motion-blur rejection)
  sfm      COLMAP run directly: GPU SIFT, exhaustive matching (or sequential + loop closure for long
           captures), incremental mapping; the largest model is kept and registration is gated
  train    nerfstudio splatfacto (3D Gaussian Splatting) on the GPU, memory-safe for 8 GB
  export   ns-export gaussian-splat -> splat.ply
  navgrid  metric scale (ArUco marker / ceiling / camera height), floor, walls -> occupancy grid,
           SOG-compressed splat and scene.json in web/public/scenes/<name>
"""
import argparse
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOCAB_URL = 'https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words32K.bin'  # FLANN format, COLMAP <= 3.11
EXHAUSTIVE_MAX = 450  # frames; exhaustive matching closes every loop and still takes minutes on a GPU at this size
STEPS = ['frames', 'sfm', 'train', 'export', 'navgrid']
IMAGE_EXT = {'.jpg', '.jpeg', '.png'}


HINTS = {
    'ffmpeg': 'Is ffmpeg installed and on PATH? Is the video file readable?',
    'colmap': 'COLMAP failed. Check that colmap is on PATH (docs/SETUP.md). If it ran, the capture likely has too much '
              'blur or blank walls: re-shoot slower with more overlap (docs/SOP.md).',
    'ns-process-data': 'Converting the COLMAP model for training failed. Re-run with --force sfm.',
    'ns-train': 'Training failed. Out of GPU memory? Retry with --max-side 1280 or --quality fast.',
    'ns-export': 'Export failed. Re-run with --force export.',
}


def run(cmd):
    print('  $', ' '.join(map(str, cmd)), flush=True)
    t = time.time()
    # resolve via PATH + PATHEXT: on Windows COLMAP ships as COLMAP.bat, which CreateProcess won't find by bare name
    exe = shutil.which(str(cmd[0])) or str(cmd[0])
    try:
        subprocess.run([exe] + [str(c) for c in cmd[1:]], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        tool = Path(str(cmd[0])).stem
        sys.exit(f'\nFAILED: {tool} ({e.__class__.__name__}). {HINTS.get(tool, "")}')
    print(f'  ({time.time() - t:.0f}s)')


def sharpness(path):
    """Variance of the Laplacian on a 480 px greyscale thumbnail: low = blurry."""
    from PIL import Image
    im = Image.open(path).convert('L')
    im.thumbnail((480, 480))
    a = np.asarray(im, dtype=np.float32)
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def step_frames(inputs, out, fps, max_frames, max_side):
    out.mkdir(parents=True, exist_ok=True)
    raw = out.parent / 'frames_raw'
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir()
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    photos = set()  # clip prefixes that are photo folders: every photo counts, no 1-in-N windowing
    for i, src in enumerate(inputs):
        if src.is_dir():
            photos.add(f'c{i:02d}')
            for j, img in enumerate(sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)):
                shutil.copy(img, raw / f'c{i:02d}_{j:05d}{img.suffix.lower()}')
        else:
            run(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'fps={fps},{scale}', '-q:v', '2', raw / f'c{i:02d}_%05d.jpg'])

    # per clip, keep the sharpest frame in each window so the kept set stays evenly spread
    frames = sorted(raw.iterdir())
    clips = {}
    for f in frames:
        clips.setdefault(f.name[:3], []).append(f)
    over = -(-len(frames) // max_frames)  # ceil: how much we must thin out to stay under max_frames
    window = max(2, over)
    kept = []
    for prefix, clip in clips.items():
        scores = [sharpness(f) for f in clip]
        med = float(np.median(scores))
        w = max(1, over) if prefix in photos else window
        for k in range(0, len(clip), w):
            best = max(range(k, min(k + w, len(clip))), key=scores.__getitem__)
            if scores[best] > 0.35 * med:  # whole window blurry (fast pan) -> drop it
                kept.append(clip[best])
    for f in kept:
        shutil.copy(f, out / f.name)
    shutil.rmtree(raw)
    print(f'  kept {len(kept)} sharp frames of {len(frames)} (window {window})')


def registered_images(model):
    """Number of registered images in a COLMAP binary model (first uint64 of images.bin)."""
    with open(model / 'images.bin', 'rb') as f:
        return struct.unpack('<Q', f.read(8))[0]


def vocab_tree(work_root):
    path = work_root / VOCAB_URL.rsplit('/', 1)[-1]
    if not path.exists():
        print(f'  downloading COLMAP vocabulary tree for loop closure ({VOCAB_URL})')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(VOCAB_URL, tmp)
        tmp.replace(path)
    return path


def step_sfm(frames, ns, matching, work_root):
    """COLMAP run directly: nerfstudio's wrapper disables loop detection and only ever reads sparse/0."""
    cm = ns / 'colmap'
    shutil.rmtree(ns, ignore_errors=True)
    (cm / 'sparse').mkdir(parents=True)
    db = cm / 'database.db'
    n_frames = sum(1 for _ in frames.iterdir())
    run(['colmap', 'feature_extractor', '--database_path', db, '--image_path', frames,
         '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'OPENCV',
         '--SiftExtraction.use_gpu', '1', '--SiftExtraction.max_num_features', '8192'])
    if matching == 'auto':
        matching = 'exhaustive' if n_frames <= EXHAUSTIVE_MAX else 'sequential'
    print(f'  matching: {matching} ({n_frames} frames)', flush=True)
    if matching == 'exhaustive':
        run(['colmap', 'exhaustive_matcher', '--database_path', db, '--SiftMatching.use_gpu', '1'])
    else:
        run(['colmap', 'sequential_matcher', '--database_path', db, '--SiftMatching.use_gpu', '1',
             '--SequentialMatching.overlap', '20', '--SequentialMatching.quadratic_overlap', '1',
             '--SequentialMatching.loop_detection', '1', '--SequentialMatching.vocab_tree_path', vocab_tree(work_root)])
    run(['colmap', 'mapper', '--database_path', db, '--image_path', frames, '--output_path', cm / 'sparse'])

    models = sorted((registered_images(m), m) for m in (cm / 'sparse').iterdir() if (m / 'images.bin').exists())
    if not models:
        sys.exit('\nFAILED: COLMAP produced no model. ' + HINTS['colmap'])
    best_n, best = models[-1]
    if best.name != '0':  # nerfstudio reads sparse/0: make that the largest model
        tmp = best.with_name('_best')
        best.rename(tmp)
        (cm / 'sparse' / '0').rename(cm / 'sparse' / str(len(models)))
        tmp.rename(cm / 'sparse' / '0')
    ratio = best_n / n_frames
    print(f'  registered {best_n}/{n_frames} frames ({ratio:.0%}) in the largest of {len(models)} model(s)')
    if len(models) > 1:
        print(f'  WARNING: the capture split into {len(models)} pieces, so some rooms are missing. Walk through doorways '
              'slowly with overlap (docs/SOP.md).')
    if ratio < 0.5:
        sys.exit(f'\nFAILED: only {ratio:.0%} of frames registered. ' + HINTS['colmap'])
    run(['ns-process-data', 'images', '--data', frames, '--output-dir', ns, '--skip-colmap',
         '--colmap-model-path', 'colmap/sparse/0', '--num-downscales', '1'])


def latest_config(outputs):
    cfgs = sorted(outputs.glob('**/config.yml'), key=lambda p: p.stat().st_mtime)
    if not cfgs:
        sys.exit(f'no trained model found under {outputs}')
    return cfgs[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', type=Path, help='video files and/or folders of photos (room-by-room clips are fine)')
    ap.add_argument('--name', required=True, help='scene id, viewer URL becomes ?scene=<name>')
    ap.add_argument('--title', help='display name in the viewer')
    ap.add_argument('--work', type=Path, default=ROOT / 'work', help='scratch dir for frames, COLMAP, checkpoints')
    ap.add_argument('--ceiling', type=float, help='tape-measured floor-to-ceiling height (m) for metric scale')
    ap.add_argument('--camera-height', type=float, default=1.4, help='phone height held during capture (m)')
    ap.add_argument('--fps', type=float, default=4, help='frames sampled per second of video before sharpness filtering')
    ap.add_argument('--max-frames', type=int, default=400)
    ap.add_argument('--max-side', type=int, default=1600, help='longest image side in px (VRAM / speed)')
    ap.add_argument('--matching', choices=['auto', 'exhaustive', 'sequential'], default='auto',
                    help=f'COLMAP matching: auto = exhaustive up to {EXHAUSTIVE_MAX} frames, else sequential + loop closure')
    ap.add_argument('--marker-size', type=float, default=0.15,
                    help='printed ArUco marker side in metres (docs/marker.svg); gives metric scale when filmed')
    ap.add_argument('--quality', choices=['fast', 'default', 'high'], default='default',
                    help='fast = 7k-iteration preview, default = 30k, high = 45k (all fit in 8 GB VRAM)')
    ap.add_argument('--force', choices=STEPS, help='re-run from this step onward')
    a = ap.parse_args()

    for p in a.inputs:
        if not p.exists():
            sys.exit(f'input not found: {p}')
    work = a.work / a.name
    frames, ns, outputs, export = work / 'frames', work / 'colmap', work / 'outputs', work / 'export'
    scene_dir = ROOT / 'web' / 'public' / 'scenes' / a.name
    redo = set(STEPS[STEPS.index(a.force):]) if a.force else set()
    todo = lambda step, done: step in redo or not done

    print('[1/5] frames')
    if todo('frames', frames.exists() and any(frames.iterdir())):
        shutil.rmtree(frames, ignore_errors=True)
        step_frames(a.inputs, frames, a.fps, a.max_frames, a.max_side)

    print('[2/5] sfm (COLMAP)')
    if todo('sfm', (ns / 'transforms.json').exists()):
        step_sfm(frames, ns, a.matching, a.work)

    print('[3/5] train (Gaussian splatting)')
    if todo('train', any(outputs.glob('**/config.yml'))):
        shutil.rmtree(outputs, ignore_errors=True)
        iters = {'fast': 7000, 'default': 30000, 'high': 45000}[a.quality]
        # splatfacto-big needs ~12 GB; these settings keep a multi-room scene inside 8 GB
        run(['ns-train', 'splatfacto', '--data', ns, '--output-dir', outputs, '--max-num-iterations', iters,
             '--viewer.quit-on-train-completion', 'True', '--vis', 'tensorboard',
             '--pipeline.datamanager.cache-images', 'cpu', '--pipeline.datamanager.cache-images-type', 'uint8',
             '--pipeline.model.use-bilateral-grid', 'True',  # per-image exposure / white balance across rooms
             '--pipeline.model.use-scale-regularization', 'True',  # fewer needle-like splats
             '--pipeline.model.camera-optimizer.mode', 'SO3xR3'])  # refine COLMAP poses during training

    cfg = latest_config(outputs)
    print('[4/5] export')
    if todo('export', (export / 'splat.ply').exists()):
        run(['ns-export', 'gaussian-splat', '--load-config', cfg, '--output-dir', export])

    print('[5/5] navgrid')
    run([sys.executable, Path(__file__).parent / 'navgrid.py', export / 'splat.ply', '--out', scene_dir,
         '--name', a.title or a.name, '--transforms', ns / 'transforms.json',
         '--dataparser', cfg.parent / 'dataparser_transforms.json', '--camera-height', a.camera_height,
         '--marker-size', a.marker_size, *(['--ceiling', a.ceiling] if a.ceiling else [])])
    print(f'\nDone. Open the viewer:  cd web && npm run dev  ->  http://localhost:5173/?scene={a.name}')


if __name__ == '__main__':
    main()
