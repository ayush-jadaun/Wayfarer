"""Depth-based free-space carving for the occupancy grid (needs torch + gsplat, i.e. the training venv).

Render the trained splat's expected depth from each training camera, then march each pixel ray through
the grid: cells a camera saw *through* inside the obstacle height band are evidence of free space
(floaters there are fake), and cells where rays *end* are evidence of real surfaces.
"""
import json
from pathlib import Path

import numpy as np


def load_views(transforms_json, dataparser_json, max_views=160):
    """Camera intrinsics + OpenCV camera-to-world matrices in the splat (dataparser) frame."""
    tf = json.loads(Path(transforms_json).read_text())
    dp = json.loads(Path(dataparser_json).read_text())
    T, s = np.array(dp['transform']), dp['scale']
    frames = sorted(tf['frames'], key=lambda f: f['file_path'])
    frames = frames[::max(1, len(frames) // max_views)]
    views = []
    for f in frames:
        c2w = np.array(f['transform_matrix']) @ np.diag([1, -1, -1, 1])  # OpenGL -> OpenCV axes
        m = np.eye(4)
        m[:3, :3] = T[:, :3] @ c2w[:3, :3]
        m[:3, 3] = s * (T[:, :3] @ c2w[:3, 3] + T[:, 3])
        views.append(m)
    K = np.array([[tf['fl_x'], 0, tf['cx']], [0, tf['fl_y'], tf['cy']], [0, 0, 1]])
    return np.array(views), K, int(tf['w']), int(tf['h'])


def render_depths(v, views, K, w, h, width=192):
    """Expected depth + alpha per view at low resolution: [V, h', w'] each."""
    import torch
    from gsplat import rasterization
    dev = 'cuda'
    sc = width / w
    hh = max(8, round(h * sc))
    Ks = torch.tensor(K * np.array([[sc], [sc], [1]]), dtype=torch.float32, device=dev)
    means = torch.tensor(np.stack([v['x'], v['y'], v['z']], 1), dtype=torch.float32, device=dev)
    quats = torch.nn.functional.normalize(torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1),
                                                       dtype=torch.float32, device=dev), dim=-1)
    scales = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1), dtype=torch.float32, device=dev))
    opac = torch.sigmoid(torch.tensor(v['opacity'], dtype=torch.float32, device=dev))
    cols = torch.zeros(len(v), 3, device=dev)
    depths, alphas = [], []
    with torch.no_grad():
        for i in range(0, len(views), 16):
            vm = torch.linalg.inv(torch.tensor(views[i:i + 16], dtype=torch.float32, device=dev))
            img, alpha, _ = rasterization(means, quats, scales, opac, cols, vm, Ks[None].expand(len(vm), 3, 3),
                                          width, hh, render_mode='ED')
            depths.append(img[..., 0].cpu().numpy())
            alphas.append(alpha[..., 0].cpu().numpy())
    return np.concatenate(depths), np.concatenate(alphas), Ks.cpu().numpy()


def carve(v, views, K, w, h, to_metric, grid, band, step=0.05, min_alpha=0.6, margin=0.12):
    """(pass_count, hit_count) grids. to_metric maps splat-frame points [N,3] to aligned metric (y up)."""
    x0, z0, cell, W, H = grid
    lo, hi = band
    depths, alphas, Ks = render_depths(v, views, K, w, h)
    V, hh, ww = depths.shape
    ys, xs = np.mgrid[0:hh:2, 0:ww:2]  # every other pixel is plenty at 10 cm cells
    rays_cam = np.stack([(xs + 0.5 - Ks[0, 2]) / Ks[0, 0], (ys + 0.5 - Ks[1, 2]) / Ks[1, 1], np.ones_like(xs, float)], -1).reshape(-1, 3)
    passed, hits = np.zeros(H * W), np.zeros(H * W)
    for i in range(V):
        d = depths[i, ::2, ::2].reshape(-1)
        ok = (alphas[i, ::2, ::2].reshape(-1) > min_alpha) & (d > 0)
        if not ok.any():
            continue
        R, c = views[i][:3, :3], views[i][:3, 3]
        dirs = rays_cam[ok] @ R.T  # z-depth -> world offset per unit depth
        end = c + dirs * d[ok, None]
        cm, em = to_metric(c[None])[0], to_metric(end)
        seg = em - cm
        length = np.linalg.norm(seg, axis=1)
        keep = length > margin + step
        if not keep.any():
            continue
        seg, length, em = seg[keep], length[keep], em[keep]
        # surface hits
        inb = (em[:, 1] > lo) & (em[:, 1] < hi)
        hx, hz = ((em[inb, 0] - x0) / cell).astype(int), ((em[inb, 2] - z0) / cell).astype(int)
        m = (hx >= 0) & (hx < W) & (hz >= 0) & (hz < H)
        np.add.at(hits, hz[m] * W + hx[m], 1)
        # free space along each ray, stopping `margin` short of the surface
        n = int(np.ceil(length.max() / step))
        t = np.arange(1, n + 1) * step
        for j in range(0, len(seg), 4096):
            sj, lj = seg[j:j + 4096], length[j:j + 4096]
            frac = t[None, :] / lj[:, None]
            valid = t[None, :] < (lj[:, None] - margin)
            pts = cm + sj[:, None, :] * frac[..., None]
            valid &= (pts[..., 1] > lo) & (pts[..., 1] < hi)
            px, pz = ((pts[..., 0] - x0) / cell).astype(int), ((pts[..., 2] - z0) / cell).astype(int)
            valid &= (px >= 0) & (px < W) & (pz >= 0) & (pz < H)
            np.add.at(passed, (pz * W + px)[valid], 1)
    return passed.reshape(H, W), hits.reshape(H, W)
