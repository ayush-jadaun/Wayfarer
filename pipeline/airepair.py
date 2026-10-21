"""AI view repair (Difix3D+ approach): fix the angles the camera never saw, then bake the fix into the 3D.

A splat only looks right from angles the capture covered. For a big venue you cannot film every object
from every side, so:
  1. synthesise novel cameras around the real ones (side steps and turns, where the twin breaks down)
  2. render them from the trained splat
  3. repair each render with NVIDIA Difix (single-step diffusion, guided by the nearest real photo)
  4. fine-tune the splat on real photos + repaired views, so every angle improves in 3D

Difix: https://github.com/nv-tlabs/Difix3D (NVIDIA License: non-commercial research/evaluation use).

  python pipeline/airepair.py --data work/home/colmap --run <splatfacto run dir> --splat work/home/export/splat.ply \
      --scene web/public/scenes/home --out work/home/airepair
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from navgrid import pose_transform, read_ply

DIFIX_SRC = Path(r'E:\tools\Difix3D\src')
DIFIX_MODEL = Path(r'E:\tools\difix_ref')  # local copy (HF cache symlinks need admin rights on Windows)
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenGL <-> OpenCV camera axes


def load_capture(data_dir, dataparser_json):
    tf = json.loads((Path(data_dir) / 'transforms.json').read_text())
    frames = sorted(tf['frames'], key=lambda f: f['file_path'])
    T, s_dp = pose_transform(Path(data_dir) / 'transforms.json', dataparser_json)
    gl = []  # camera-to-world in the splat frame (OpenGL axes), i.e. what the trained model saw
    for f in frames:
        c2w = np.array(f['transform_matrix'])[:3]
        m = np.eye(4)
        m[:3, :3] = T[:, :3] @ c2w[:, :3]
        m[:3, 3] = s_dp * (T[:, :3] @ c2w[:, 3] + T[:, 3])
        gl.append(m)
    K = np.array([[tf['fl_x'], 0, tf['cx']], [0, tf['fl_y'], tf['cy']], [0, 0, 1]])
    return tf, frames, np.array(gl), K, int(tf['w']), int(tf['h'])


def novel_poses(gl, up, metres_per_unit, n_max=150, rng=None):
    """Side steps and turns around real cameras: the views a visitor will take that the capture didn't."""
    rng = rng or np.random.default_rng(0)
    out, src = [], []
    step = max(1, len(gl) * 4 // n_max)
    for i in range(0, len(gl), step):
        c2w = gl[i]
        right = c2w[:3, 0] - (c2w[:3, 0] @ up) * up
        right /= np.linalg.norm(right) + 1e-9
        for kind in ('left', 'right', 'turn_l', 'turn_r'):
            m = c2w.copy()
            if kind in ('left', 'right'):
                m[:3, 3] += right * (0.5 / metres_per_unit) * (1 if kind == 'right' else -1)
                m[:3, :3] = Rotation.from_rotvec(up * math.radians(rng.uniform(-10, 10))).as_matrix() @ m[:3, :3]
            else:
                ang = math.radians(rng.uniform(25, 40)) * (1 if kind == 'turn_l' else -1)
                m[:3, :3] = Rotation.from_rotvec(up * ang).as_matrix() @ m[:3, :3]
            out.append(m)
            src.append(i)
    return np.array(out[:n_max]), np.array(src[:n_max])


def render_views(splat, c2ws_gl, K, w, h):
    import torch
    from gsplat import rasterization
    dev = 'cuda'
    v = splat
    n = len(v)
    f_rest = [k for k in v.dtype.names if k.startswith('f_rest_')]
    sh_deg = int(round(math.sqrt(len(f_rest) // 3 + 1))) - 1
    colors = np.concatenate([np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1)[:, None, :],
                             np.stack([v[k] for k in f_rest], 1).reshape(n, 3, -1).transpose(0, 2, 1)], 1)
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    means = t(np.stack([v['x'], v['y'], v['z']], 1))
    quats = torch.nn.functional.normalize(t(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1)), dim=-1)
    scales = torch.exp(t(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1)))
    opac = torch.sigmoid(t(v['opacity']))
    cols = t(colors)
    Kt = t(K)[None]
    imgs = []
    with torch.no_grad():
        for c2w in c2ws_gl:
            cv = c2w.copy()
            cv[:3, :3] = c2w[:3, :3] @ FLIP
            vm = torch.linalg.inv(t(cv))[None]
            img, _, _ = rasterization(means, quats, scales, opac, cols, vm, Kt, w, h, sh_degree=sh_deg,
                                      backgrounds=torch.zeros(1, 3, device=dev))
            imgs.append(Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
    del means, quats, scales, opac, cols
    torch.cuda.empty_cache()
    return imgs


def prepare_checkpoint(src_dir, dst_dir, n_train):
    """Checkpoint for fine-tuning on a dataset with a different image count.

    nerfstudio loads strictly, but two parts are per-image: the camera-pose corrections (dropped: the AI
    views have exact poses) and the bilateral exposure grids (resized to the new training-image count,
    initialised to the average grid of the original images). The Gaussians and their optimiser state
    carry over untouched."""
    import torch
    src = sorted(Path(src_dir).glob('step-*.ckpt'))[-1]
    state = torch.load(src, map_location='cpu')
    pipe = state['pipeline']
    for k in [k for k in pipe if 'camera_optimizer' in k]:
        del pipe[k]
    for k in [k for k in pipe if k.endswith('bil_grids.grids')]:
        g = pipe[k]
        pipe[k] = g.mean(0, keepdim=True).repeat(n_train, *([1] * (g.dim() - 1)))
    # fresh optimiser/scheduler state: a short fine-tune doesn't need Adam history, and it is 2/3 of the file
    state['optimizers'], state['schedulers'] = {}, {}
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(dst_dir) / src.name)
    return Path(dst_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='ns-process-data folder (transforms.json, images/)')
    ap.add_argument('--run', required=True, help='splatfacto run dir (config.yml, dataparser_transforms.json)')
    ap.add_argument('--splat', required=True, help='exported splat.ply of that run')
    ap.add_argument('--scene', required=True, help='scene folder (for metric scale and up direction)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', type=int, default=150, help='number of AI-repaired views')
    ap.add_argument('--width', type=int, default=1024, help='render width (Difix works best around 1K)')
    ap.add_argument('--finetune-steps', type=int, default=6000)
    ap.add_argument('--no-finetune', action='store_true', help='only produce repaired views')
    ap.add_argument('--skip-repair', action='store_true', help='reuse repaired views from --out and only fine-tune')
    a = ap.parse_args()

    out = Path(a.out)
    (out / 'images_ai').mkdir(parents=True, exist_ok=True)
    run = Path(a.run)
    if a.skip_repair and (out / 'transforms.json').exists():  # reuse repaired views from an earlier run
        meta = json.loads((out / 'transforms.json').read_text())
        print(f'  reusing {len(meta["frames"])} views from {out / "transforms.json"}')
        return finetune(a, out, run, meta)
    tf, frames, gl, K, W, H = load_capture(a.data, run / 'dataparser_transforms.json')
    scene = json.loads((Path(a.scene) / 'scene.json').read_text())
    tr = scene['splat']
    R_sc = Rotation.from_quat(tr['rotation']).as_matrix()
    up = R_sc.T @ np.array([0, 1.0, 0])  # metric +y expressed in the splat frame
    m_per_unit = tr['scale'][0]

    novel, src = novel_poses(gl, up, m_per_unit, a.views)
    w = a.width - a.width % 8
    h = int(round(H * w / W)) // 8 * 8
    Ks = K * np.array([[w / W], [h / H], [1]])
    print(f'  rendering {len(novel)} novel views at {w}x{h}')
    renders = render_views(read_ply(a.splat), novel, Ks, w, h)

    print('  repairing them with Difix (single-step diffusion, reference = nearest real photo)')
    import torch
    sys.path.insert(0, str(DIFIX_SRC))
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained(str(DIFIX_MODEL) if DIFIX_MODEL.exists() else 'nvidia/difix_ref',
                                         trust_remote_code=True, torch_dtype=torch.float16)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    root = Path(a.data)
    ai_frames = []
    for k, (img, c2w, s) in enumerate(zip(renders, novel, src)):
        ref = Image.open(root / frames[s]['file_path']).convert('RGB').resize((w, h))
        fixed = pipe('remove degradation', image=img, ref_image=ref, num_inference_steps=1, timesteps=[199],
                     guidance_scale=0.0).images[0].resize((w, h))
        name = f'images_ai/ai_{k:04d}.png'
        fixed.save(out / name)
        if k < 6:  # before/after pairs for the docs
            Image.fromarray(np.hstack([np.asarray(img), np.asarray(fixed)])).save(out / f'compare_{k}.jpg', quality=88)
        ai_frames.append({'file_path': str((out / name).resolve()), 'transform_matrix': c2w.tolist(),
                          'fl_x': Ks[0, 0], 'fl_y': Ks[1, 1], 'cx': Ks[0, 2], 'cy': Ks[1, 2], 'w': w, 'h': h,
                          'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0})
        if k % 25 == 0:
            print(f'    {k}/{len(renders)}', flush=True)
    del pipe
    torch.cuda.empty_cache()

    # dataset for fine-tuning, already in the trained model's frame: the dataparser must leave poses alone
    # intrinsics per frame: top-level values would make nerfstudio force 1600x900 onto the 1024x576 AI views
    intr = {k: tf[k] for k in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'k1', 'k2', 'p1', 'p2') if k in tf}
    real = [{'file_path': str((root / f['file_path']).resolve()), 'transform_matrix': m.tolist(), **intr}
            for f, m in zip(frames, gl)]
    meta = {'camera_model': tf.get('camera_model', 'OPENCV'), 'frames': real + ai_frames}
    (out / 'transforms.json').write_text(json.dumps(meta, indent=1))
    print(f'  wrote {len(real)} real + {len(ai_frames)} AI-repaired views -> {out / "transforms.json"}')
    if not a.no_finetune:
        finetune(a, out, run, meta)


def finetune(a, out, run, meta):
    # 4. bake the repairs into the 3D: continue training from the checkpoint on real + repaired views.
    # Poses are already in the model's frame, so the dataparser must not re-orient, re-centre or re-scale.
    import shutil
    import subprocess
    ns = lambda tool: shutil.which(tool) or tool
    n_train = math.ceil(len(meta['frames']) * 0.9)  # nerfstudio's default 90/10 train/eval split
    init = prepare_checkpoint(run / 'nerfstudio_models', out / 'init_ckpt', n_train)
    subprocess.run([ns('ns-train'), 'splatfacto', '--data', str(out), '--output-dir', str(out / 'outputs'),
                    '--load-dir', str(init), '--max-num-iterations', str(a.finetune_steps),
                    '--viewer.quit-on-train-completion', 'True', '--vis', 'tensorboard',
                    '--pipeline.datamanager.cache-images', 'cpu', '--pipeline.datamanager.cache-images-type', 'uint8',
                    '--pipeline.model.use-bilateral-grid', 'True', '--pipeline.model.use-scale-regularization', 'True',
                    '--pipeline.model.camera-optimizer.mode', 'off',  # AI views have exact poses by construction
                    'nerfstudio-data', '--orientation-method', 'none', '--center-method', 'none',
                    '--auto-scale-poses', 'False'], check=True)
    cfg = sorted((out / 'outputs').glob('**/config.yml'), key=lambda p: p.stat().st_mtime)[-1]
    subprocess.run([ns('ns-export'), 'gaussian-splat', '--load-config', str(cfg), '--output-dir', str(out)], check=True)
    print(f'  repaired splat -> {out / "splat.ply"} (same frame as the original model)')


if __name__ == '__main__':
    main()
