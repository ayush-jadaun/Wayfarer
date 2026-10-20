"""Wayfarer pipeline: phone video(s) / photos -> web-ready navigable digital twin.

  python pipeline/wayfarer.py capture.mp4 --name home --ceiling 2.75

Steps (each is skipped when its output already exists; --force STEP re-runs from STEP):
  frames   ffmpeg -> keep the sharpest frame per window (motion-blur rejection)
  sfm      COLMAP via ns-process-data (camera poses + sparse points)
  train    nerfstudio splatfacto (3D Gaussian Splatting) on the GPU
  export   ns-export gaussian-splat -> splat.ply
  navgrid  floor/scale/walls -> occupancy grid, compact .splat, scene.json in web/public/scenes/<name>
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STEPS = ['frames', 'sfm', 'train', 'export', 'navgrid']
VIDEO_EXT = {'.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png'}


def run(cmd):
    print('  $', ' '.join(map(str, cmd)), flush=True)
    t = time.time()
    subprocess.run([str(c) for c in cmd], check=True)
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
    for i, src in enumerate(inputs):
        if src.is_dir():
            for j, img in enumerate(sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)):
                shutil.copy(img, raw / f'c{i:02d}_{j:05d}{img.suffix.lower()}')
        else:
            run(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'fps={fps},{scale}', '-q:v', '2', raw / f'c{i:02d}_%05d.jpg'])

    # per clip, keep the sharpest frame in each window so the kept set stays evenly spread
    frames = sorted(raw.iterdir())
    clips = {}
    for f in frames:
        clips.setdefault(f.name[:3], []).append(f)
    window = max(2, -(-len(frames) // max_frames))  # ceil
    kept = []
    for clip in clips.values():
        scores = [sharpness(f) for f in clip]
        med = float(np.median(scores))
        for k in range(0, len(clip), window):
            best = max(range(k, min(k + window, len(clip))), key=scores.__getitem__)
            if scores[best] > 0.35 * med:  # whole window blurry (fast pan) -> drop it
                kept.append(clip[best])
    for f in kept:
        shutil.copy(f, out / f.name)
    shutil.rmtree(raw)
    print(f'  kept {len(kept)} sharp frames of {len(frames)} (window {window})')


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
    ap.add_argument('--matching', choices=['sequential', 'exhaustive', 'vocab_tree'],
                    help='COLMAP matching (default: sequential for one video, exhaustive for several clips)')
    ap.add_argument('--quality', choices=['fast', 'default', 'high'], default='default',
                    help='fast = 7k iters preview, default = splatfacto 30k, high = splatfacto-big')
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
        matching = a.matching or ('sequential' if len(a.inputs) == 1 and a.inputs[0].suffix.lower() in VIDEO_EXT else 'exhaustive')
        run(['ns-process-data', 'images', '--data', frames, '--output-dir', ns, '--matching-method', matching])

    print('[3/5] train (Gaussian splatting)')
    if todo('train', any(outputs.glob('**/config.yml'))):
        method = 'splatfacto-big' if a.quality == 'high' else 'splatfacto'
        iters = 7000 if a.quality == 'fast' else 30000
        run(['ns-train', method, '--data', ns, '--output-dir', outputs, '--max-num-iterations', iters,
             '--viewer.quit-on-train-completion', 'True', '--vis', 'tensorboard'])

    cfg = latest_config(outputs)
    print('[4/5] export')
    if todo('export', (export / 'splat.ply').exists()):
        run(['ns-export', 'gaussian-splat', '--load-config', cfg, '--output-dir', export])

    print('[5/5] navgrid')
    run([sys.executable, Path(__file__).parent / 'navgrid.py', export / 'splat.ply', '--out', scene_dir,
         '--name', a.title or a.name, '--transforms', ns / 'transforms.json',
         '--dataparser', cfg.parent / 'dataparser_transforms.json', '--camera-height', a.camera_height,
         *(['--ceiling', a.ceiling] if a.ceiling else [])])
    print(f'\nDone. Open the viewer:  cd web && npm run dev  ->  http://localhost:5173/?scene={a.name}')


if __name__ == '__main__':
    main()
