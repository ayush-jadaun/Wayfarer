"""Triangle mesh of the twin (needs torch + gsplat, i.e. the training venv).

nerfstudio's TSDF exporter only works with NeRFs, so this fuses the splat's own rendered depth from every
training camera into a TSDF volume on the GPU, runs marching cubes and colours vertices from the nearest
Gaussian. The mesh is in the same frame as splat.ply. Uses: collision, measurement, Blender/Unity import.

  python pipeline/mesh.py --splat work/x/export/splat.ply --transforms work/x/colmap/transforms.json \
      --dataparser work/x/outputs/.../dataparser_transforms.json --out work/x/export/mesh.ply
"""
import argparse
from pathlib import Path

import numpy as np

import depthcarve
from navgrid import read_ply, sigmoid


def surface_points(depths, alphas, Ks, views, min_alpha, stride=8):
    hh, ww = depths.shape[1:]
    ys, xs = np.mgrid[0:hh:stride, 0:ww:stride]
    rays = np.stack([(xs + 0.5 - Ks[0, 2]) / Ks[0, 0], (ys + 0.5 - Ks[1, 2]) / Ks[1, 1], np.ones_like(xs, float)], -1)
    out = []
    for d, a, m in zip(depths[:, ::stride, ::stride], alphas[:, ::stride, ::stride], views):
        ok = a > min_alpha
        out.append((rays[ok] * d[ok, None]) @ m[:3, :3].T + m[:3, 3])
    return np.concatenate(out)


def fuse(v, views, K, w, h, res=320, width=384, min_alpha=0.6):
    """TSDF volume over what the cameras saw: (tsdf [X, Y, Z], weight, origin, voxel size)."""
    import torch
    depths, alphas, Ks = depthcarve.render_depths(v, views, K, w, h, width=width)
    # expected depth blends foreground and background at silhouettes: those "flying pixels" fuse into
    # streaks along view rays, so drop pixels where depth jumps more than 3% between neighbours
    gy, gx = np.gradient(depths, axis=(1, 2))
    alphas[np.hypot(gx, gy) > 0.03 * np.maximum(depths, 1e-3)] = 0
    # extent from rendered surfaces, not Gaussians: far floaters would blow up the volume and the voxel size
    pts = surface_points(depths, alphas, Ks, views, min_alpha)
    lo, hi = np.percentile(pts, 0.5, 0), np.percentile(pts, 99.5, 0)
    vox = (hi - lo).max() / res
    lo -= 4 * vox
    dims = np.ceil((hi - lo) / vox).astype(int) + 8
    trunc = 3 * vox
    dev = 'cuda'
    axes = [torch.arange(n, device=dev, dtype=torch.float32) * vox + o for n, o in zip(dims, lo)]
    pts = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    tsdf = torch.zeros(len(pts), device=dev)
    weight = torch.zeros(len(pts), device=dev)
    fx, fy, cx, cy = Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]
    hh, ww = depths.shape[1:]
    for i, m in enumerate(views):
        R = torch.tensor(m[:3, :3], dtype=torch.float32, device=dev)
        c = torch.tensor(m[:3, 3], dtype=torch.float32, device=dev)
        pc = (pts - c) @ R  # world -> OpenCV camera
        z = pc[:, 2]
        u = (pc[:, 0] / z.clamp(min=1e-6) * fx + cx).long()
        r = (pc[:, 1] / z.clamp(min=1e-6) * fy + cy).long()
        ok = (z > 0) & (u >= 0) & (u < ww) & (r >= 0) & (r < hh)
        d = torch.tensor(depths[i], device=dev)[r.clamp(0, hh - 1), u.clamp(0, ww - 1)]
        a = torch.tensor(alphas[i], device=dev)[r.clamp(0, hh - 1), u.clamp(0, ww - 1)]
        sdf = d - z
        ok &= (a > min_alpha) & (d > 0) & (sdf > -trunc)  # behind the surface by more than trunc: unseen
        s = (sdf[ok] / trunc).clamp(max=1)
        tsdf[ok] = (tsdf[ok] * weight[ok] + s) / (weight[ok] + 1)
        weight[ok] += 1
    return tsdf.reshape(*dims).cpu().numpy(), weight.reshape(*dims).cpu().numpy(), lo, vox


def build_mesh(v, views, K, w, h, out, target_faces=400_000, min_views=3):
    import pymeshlab
    from scipy.spatial import cKDTree
    from skimage import measure
    tsdf, weight, lo, vox = fuse(v, views, K, w, h)
    verts, faces, _, _ = measure.marching_cubes(tsdf, 0, mask=weight >= min_views)
    verts = verts * vox + lo
    means = np.stack([v['x'], v['y'], v['z']], 1)
    keep = sigmoid(v['opacity']) > 0.3
    _, idx = cKDTree(means[keep]).query(verts)
    dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1)[keep][idx]
    rgb = np.clip(0.5 + 0.28209479 * dc, 0, 1)  # SH degree-0 -> colour
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts, faces, v_color_matrix=np.c_[rgb, np.ones(len(rgb))]))
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(2))
    if ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservenormal=True)
    ms.save_current_mesh(str(out))
    return ms.current_mesh().vertex_number(), ms.current_mesh().face_number()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splat', type=Path, required=True)
    ap.add_argument('--transforms', type=Path, required=True)
    ap.add_argument('--dataparser', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--faces', type=int, default=400_000)
    a = ap.parse_args()
    views = depthcarve.load_views(a.transforms, a.dataparser, max_views=10 ** 9)
    nv, nf = build_mesh(read_ply(a.splat), *views, a.out, a.faces)
    print(f'mesh: {nv} vertices, {nf} faces -> {a.out}')


if __name__ == '__main__':
    main()
