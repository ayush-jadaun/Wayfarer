"""3D Gaussian Splatting with MCMC densification (gsplat MCMCStrategy), sized for an 8 GB GPU.

nerfstudio 1.1.5 has no MCMC option and gsplat's example trainer needs CUDA extensions compiled from
source (fused-ssim) that fail on Windows; this uses only the prebuilt gsplat kernels. MCMC treats
Gaussians as samples and relocates dead ones instead of cloning/splitting, so a hard cap (default 1M)
bounds memory and it produces fewer floaters at equal quality.

  python pipeline/train_mcmc.py --data work/home/colmap --out work/home/outputs/mcmc --steps 30000

Inputs: an ns-process-data folder (transforms.json, images/, sparse_pc.ply).
Outputs in --out: splat.ply (INRIA/nerfstudio layout), dataparser_transforms.json, metrics.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814


# ---------------------------------------------------------------- data
def load_dataset(root, holdout_every=10, device='cuda'):
    import cv2
    root = Path(root)
    tf = json.loads((root / 'transforms.json').read_text())
    frames = sorted(tf['frames'], key=lambda f: f['file_path'])
    K = np.array([[tf['fl_x'], 0, tf['cx']], [0, tf['fl_y'], tf['cy']], [0, 0, 1]])
    dist = np.array([tf.get(k, 0.0) for k in ('k1', 'k2', 'p1', 'p2')])
    imgs, c2ws = [], []
    for f in frames:
        img = cv2.cvtColor(cv2.imread(str(root / f['file_path'])), cv2.COLOR_BGR2RGB)
        if np.any(dist):
            img = cv2.undistort(img, K, dist)  # train a pinhole model on undistorted images
        imgs.append(torch.from_numpy(img))  # uint8 on CPU: ~1/4 the memory of float on GPU
        c2ws.append(np.array(f['transform_matrix']))
    c2ws = np.array(c2ws)

    # normalise like nerfstudio's dataparser: average camera up -> +z, centre the cameras, fit in [-1, 1]
    up = c2ws[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    v = np.cross(up, [0, 0, 1.0])
    s, c = np.linalg.norm(v), up @ [0, 0, 1.0]
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s)) if s > 1e-8 else np.eye(3) * np.sign(c)
    center = c2ws[:, :3, 3].mean(0)
    T = np.concatenate([R, (-R @ center)[:, None]], 1)
    pos = c2ws[:, :3, 3] @ R.T + T[:, 3]
    scale = float(1.0 / np.abs(pos).max())
    c2w_n = np.zeros_like(c2ws)
    c2w_n[:, :3, :3] = R @ c2ws[:, :3, :3]
    c2w_n[:, :3, 3] = pos * scale
    c2w_n[:, 3, 3] = 1
    viewmats = torch.linalg.inv(torch.tensor(c2w_n @ np.diag([1, -1, -1, 1]), dtype=torch.float32))  # OpenGL -> OpenCV

    pts, cols = read_points(root / tf.get('ply_file_path', 'sparse_pc.ply'))
    pts = (pts @ R.T + T[:, 3]) * scale
    idx = np.arange(len(frames))
    test = idx[::holdout_every] if len(frames) >= 20 else idx[:0]
    train = np.setdiff1d(idx, test)
    return dict(imgs=imgs, viewmats=viewmats, K=torch.tensor(K, dtype=torch.float32), w=int(tf['w']), h=int(tf['h']),
                points=pts, colors=cols, train=train, test=test, transform=T, scale=scale,
                extent=float(np.linalg.norm(c2w_n[:, :3, 3] - c2w_n[:, :3, 3].mean(0), axis=1).max()))


def read_points(path):
    """sparse_pc.ply from ns-process-data (binary or ascii, xyz + rgb)."""
    with open(path, 'rb') as f:
        header = []
        while (line := f.readline().decode('ascii').strip()) != 'end_header':
            header.append(line)
        n = int(next(l.split()[2] for l in header if l.startswith('element vertex')))
        types = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8', 'uchar': 'u1', 'uint8': 'u1'}
        props = [(l.split()[2], types[l.split()[1]]) for l in header if l.startswith('property')]
        if any('ascii' in l for l in header):
            data = np.loadtxt(f, max_rows=n)
            arr = {name: data[:, i] for i, (name, _) in enumerate(props)}
        else:
            arr = np.fromfile(f, dtype=np.dtype(props), count=n)
    pts = np.stack([arr['x'], arr['y'], arr['z']], 1).astype(np.float64)
    if 'red' in [name for name, _ in props]:
        cols = np.stack([arr['red'], arr['green'], arr['blue']], 1).astype(np.float32) / 255
    else:
        cols = np.full((n, 3), 0.5, np.float32)
    return pts, cols


# ---------------------------------------------------------------- losses
def ssim(a, b, window=11, sigma=1.5):
    """Structural similarity on [1, 3, H, W] images (Gaussian window)."""
    g = torch.exp(-(torch.arange(window, device=a.device) - window // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, window, window)
    conv = lambda x: F.conv2d(x, w, padding=window // 2, groups=3)
    mu_a, mu_b = conv(a), conv(b)
    va, vb, cov = conv(a * a) - mu_a ** 2, conv(b * b) - mu_b ** 2, conv(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean()


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b)))


# ---------------------------------------------------------------- model
def init_params(points, colors, extent, sh_degree=3, device='cuda'):
    p = torch.tensor(points, dtype=torch.float32, device=device)
    if len(p) < 1000:  # too few SfM points: pad with random ones inside the camera volume
        p = torch.cat([p, (torch.rand(20000, 3, device=device) * 2 - 1) * extent * 1.5])
        colors = np.concatenate([colors, np.full((20000, 3), 0.5, np.float32)])
    # initial size = mean distance to the 3 nearest neighbours
    d = torch.cat([torch.cdist(p[i:i + 4096], p).topk(4, largest=False).values[:, 1:] for i in range(0, len(p), 4096)])
    dist = d.mean(1).clamp_min(1e-7)
    n = len(p)
    rgb = torch.tensor(colors, dtype=torch.float32, device=device)
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(p),
        'scales': torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        'quats': torch.nn.Parameter(F.normalize(torch.rand(n, 4, device=device), dim=-1)),
        'opacities': torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=device))),
        'sh0': torch.nn.Parameter(((rgb - 0.5) / C0)[:, None, :]),
        'shN': torch.nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3, device=device)),
    })
    return params


def render(params, viewmat, K, w, h, sh_degree):
    from gsplat import rasterization
    colors = torch.cat([params['sh0'], params['shN']], 1)
    img, alpha, info = rasterization(params['means'], params['quats'], torch.exp(params['scales']),
                                     torch.sigmoid(params['opacities']), colors, viewmat[None], K[None], w, h,
                                     sh_degree=sh_degree, packed=False)
    return img[0], info


def save_ply(params, path):
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = len(p['means'])
    f_rest = p['shN'].transpose(0, 2, 1).reshape(n, -1)  # channel-major, as in the INRIA format
    cols = ([('x', p['means'][:, 0]), ('y', p['means'][:, 1]), ('z', p['means'][:, 2])]
            + [(f'n{a}', np.zeros(n, np.float32)) for a in 'xyz']
            + [(f'f_dc_{i}', p['sh0'][:, 0, i]) for i in range(3)]
            + [(f'f_rest_{i}', f_rest[:, i]) for i in range(f_rest.shape[1])]
            + [('opacity', p['opacities'])]
            + [(f'scale_{i}', p['scales'][:, i]) for i in range(3)]
            + [(f'rot_{i}', p['quats'][:, i]) for i in range(4)])
    arr = np.empty(n, dtype=[(name, '<f4') for name, _ in cols])
    for name, val in cols:
        arr[name] = val
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n
                 + ''.join(f'property float {name}\n' for name, _ in cols) + 'end_header\n').encode())
        arr.tofile(f)


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--cap-max', type=int, default=1_000_000, help='Gaussian budget (VRAM bound)')
    ap.add_argument('--sh-degree', type=int, default=3)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.data)
    print(f'{len(ds["train"])} train / {len(ds["test"])} held-out views, {len(ds["points"])} SfM points', flush=True)
    params = init_params(ds['points'], ds['colors'], ds['extent'], a.sh_degree)
    K, w, h = ds['K'].to(dev), ds['w'], ds['h']
    viewmats = ds['viewmats'].to(dev)
    n_img = len(ds['imgs'])
    # per-image affine colour correction (exposure / white balance differs between rooms)
    color_fix = torch.nn.Parameter(torch.eye(3, 4, device=dev)[None].repeat(n_img, 1, 1))

    from gsplat.strategy import MCMCStrategy
    lrs = {'means': 1.6e-4 * ds['extent'], 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    opt_fix = torch.optim.Adam([color_fix], lr=1e-3)
    strategy = MCMCStrategy(cap_max=a.cap_max, refine_stop_iter=int(a.steps * 0.83))
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state()
    decay = 0.01 ** (1 / a.steps)  # means lr decays to 1% over training

    imgs_gpu = {}
    t0 = time.time()
    order = np.random.default_rng(0).permutation(np.tile(ds['train'], a.steps // len(ds['train']) + 1))
    for step in range(a.steps):
        i = int(order[step])
        if i not in imgs_gpu:  # cache on GPU lazily as uint8 (small), convert per step
            imgs_gpu[i] = ds['imgs'][i].to(dev)
        gt = imgs_gpu[i].float() / 255
        sh = min(step // 1000, a.sh_degree)
        pred, info = render(params, viewmats[i], K, w, h, sh)
        pred = pred @ color_fix[i, :, :3].T + color_fix[i, :, 3]
        strategy.step_pre_backward(params, opts, state, step, info)
        l1 = (pred - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        loss = loss + 0.01 * torch.sigmoid(params['opacities']).mean() + 0.01 * torch.exp(params['scales']).mean()
        loss.backward()
        for o in [*opts.values(), opt_fix]:
            o.step()
            o.zero_grad(set_to_none=True)
        for g in opts['means'].param_groups:
            g['lr'] *= decay
        strategy.step_post_backward(params, opts, state, step, info, lr=opts['means'].param_groups[0]['lr'])
        if step % 500 == 0 or step == a.steps - 1:
            el = time.time() - t0
            eta = el / (step + 1) * (a.steps - step - 1)
            print(f'step {step:6d}/{a.steps}  loss {loss.item():.4f}  gaussians {len(params["means"]):,}  '
                  f'{el / 60:.1f} min, eta {eta / 60:.1f} min', flush=True)

    # held-out evaluation (no colour correction: fair against nerfstudio's numbers)
    metrics = {}
    if len(ds['test']):
        ps, ss = [], []
        with torch.no_grad():
            for i in ds['test']:
                gt = ds['imgs'][i].to(dev).float() / 255
                pred, _ = render(params, viewmats[i], K, w, h, a.sh_degree)
                pred = pred.clamp(0, 1)
                ps.append(psnr(pred, gt))
                ss.append(float(ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
        metrics = {'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss))}
        print(f'held-out views: PSNR {metrics["psnr"]:.2f} dB, SSIM {metrics["ssim"]:.3f}')
    (out / 'metrics.json').write_text(json.dumps({'method_name': 'mcmc', 'results': metrics}, indent=1))
    # nerfstudio convention: the saved transform maps *original* data coordinates, i.e. it includes applied_transform
    T = np.vstack([ds['transform'], [0, 0, 0, 1]])
    meta = json.loads((Path(a.data) / 'transforms.json').read_text())
    if 'applied_transform' in meta:
        T = T @ np.vstack([np.array(meta['applied_transform']), [0, 0, 0, 1]])
    (out / 'dataparser_transforms.json').write_text(json.dumps({'transform': T[:3].tolist(), 'scale': ds['scale']}))
    save_ply(params, out / 'splat.ply')
    print(f'saved {len(params["means"]):,} Gaussians -> {out / "splat.ply"} ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
