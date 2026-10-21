"""Fill floor and ceiling holes with real texture projected from the capture frames.

Captures filmed at chest height barely see the floor and ceiling, so the splat has holes there. We know
both planes exactly (floor at y=0, ceiling at the measured height), so for every 2 cm texel we gather
the frames that saw it -- occlusion-checked against the splat's rendered depth so a wall or bed can't
paint onto the floor behind it -- and blend their real pixels, favouring close, head-on views. Texels
no frame saw are inpainted from their neighbours.

  python pipeline/surfaces.py --scene web/public/scenes/home --splat work/home/export/splat.ply \
      --transforms work/home/colmap/transforms.json --dataparser <run>/dataparser_transforms.json --ceiling 2.8
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from navgrid import dilate, pose_transform, read_ply

TEXEL = 0.02  # metres per texel


def in_fov(pc, K, w, h, margin=0.98):
    """Points inside the camera's undistorted field of view. Must be checked BEFORE distortion: the lens
    polynomial folds points far outside the frame back into the image, painting wrapped garbage."""
    nx, ny = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    return ((nx > -K[0, 2] / K[0, 0] * margin) & (nx < (w - K[0, 2]) / K[0, 0] * margin)
            & (ny > -K[1, 2] / K[1, 1] * margin) & (ny < (h - K[1, 2]) / K[1, 1] * margin))
MAX_SIDE = 2048


def metric_cameras(transforms_json, dataparser_json, scene):
    """OpenCV world-to-camera matrices in the scene's metric frame, plus intrinsics and image paths."""
    tf = json.loads(Path(transforms_json).read_text())
    frames = sorted(tf['frames'], key=lambda f: f['file_path'])
    T, s_dp = pose_transform(transforms_json, dataparser_json)
    tr = scene['splat']
    R_sc, s_sc, t_sc = Rotation.from_quat(tr['rotation']).as_matrix(), tr['scale'][0], np.array(tr['position'])
    w2cs = []
    for f in frames:
        c2w = np.array(f['transform_matrix'])[:3] @ np.diag([1, -1, -1, 1])  # OpenGL -> OpenCV camera axes
        rot = R_sc @ T[:, :3] @ c2w[:, :3]
        pos = s_sc * (R_sc @ (s_dp * (T[:, :3] @ c2w[:, 3] + T[:, 3]))) + t_sc
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = rot, pos
        w2cs.append(np.linalg.inv(m))
    K = np.array([[tf['fl_x'], 0, tf['cx']], [0, tf['fl_y'], tf['cy']], [0, 0, 1]])
    dist = np.array([tf.get(k, 0.0) for k in ('k1', 'k2', 'p1', 'p2')])
    root = Path(transforms_json).parent
    return np.array(w2cs), K, dist, [root / f['file_path'] for f in frames], int(tf['w']), int(tf['h'])


def project_plane(y, normal_y, region, origin, cell, cams, depth_check, log):
    """Blend camera pixels onto the horizontal plane at height y. Returns (rgb float, observed mask)."""
    w2cs, K, dist, paths, W_img, H_img = cams
    H_c, W_c = region.shape
    step = TEXEL
    tw, th = int(round(W_c * cell / step)), int(round(H_c * cell / step))
    scale_down = max(1, max(tw, th) / MAX_SIDE)
    step *= scale_down
    tw, th = int(round(W_c * cell / step)), int(round(H_c * cell / step))
    # texel centres in metric world coordinates, only inside the mapped region
    xs = origin[0] + (np.arange(tw) + 0.5) * step
    zs = origin[1] + (np.arange(th) + 0.5) * step
    gx, gz = np.meshgrid(xs, zs)
    inside = region[np.clip((gz - origin[1]) / cell, 0, H_c - 1).astype(int), np.clip((gx - origin[0]) / cell, 0, W_c - 1).astype(int)]
    idx = np.flatnonzero(inside)
    P = np.stack([gx.ravel()[idx], np.full(len(idx), y), gz.ravel()[idx]], 1)
    acc, wsum = np.zeros((len(idx), 3)), np.zeros(len(idx))
    for i, (w2c, path) in enumerate(zip(w2cs, paths)):
        pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        front = pc[:, 2] > 0.2
        if not front.any():
            continue
        cam_pos = -w2c[:3, :3].T @ w2c[:3, 3]
        ray = P - cam_pos
        dist_m = np.linalg.norm(ray, axis=1)
        cos_inc = np.abs(ray[:, 1]) / np.maximum(dist_m, 1e-6)  # 1 = looking straight at the plane
        facing = (np.sign(cam_pos[1] - y) == normal_y)          # camera is on the visible side
        ok = front & facing & (cos_inc > 0.55) & (dist_m < 4.0)  # steep views only: walls can't bleed onto the floor
        ok[ok] = in_fov(pc[ok], K, W_img, H_img)
        if not ok.any():
            continue
        uv, _ = cv2.projectPoints(pc[ok].astype(np.float64), np.zeros(3), np.zeros(3), K, dist)  # camera coords in
        uv = uv.reshape(-1, 2)
        inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W_img - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H_img - 1)
        sel = np.flatnonzero(ok)[inimg]
        if not len(sel):
            continue
        uv = uv[inimg]
        vis = depth_check(i, uv, pc[sel, 2])
        sel, uv = sel[vis], uv[vis]
        if not len(sel):
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = np.concatenate([  # remap's map must stay under 32767 rows
            cv2.remap(img, uv[j:j + 30000, 0].astype(np.float32)[:, None], uv[j:j + 30000, 1].astype(np.float32)[:, None],
                      cv2.INTER_LINEAR)[:, 0, ::-1] for j in range(0, len(uv), 30000)]).astype(np.float64)
        w = cos_inc[sel] ** 6 / np.maximum(dist_m[sel], 0.5) ** 4  # best head-on close views dominate: no reflection smear
        acc[sel] += rgb * w[:, None]
        wsum[sel] += w
        if i % 50 == 0:
            log(f'    {i}/{len(paths)} frames')
    out = np.zeros((th * tw, 3))
    seen = np.zeros(th * tw, bool)
    good = wsum > 0
    out[idx[good]] = acc[good] / wsum[good, None]
    seen[idx[good]] = True
    return out.reshape(th, tw, 3), seen.reshape(th, tw), inside, step


_lama = None


def generative_fill(img, hole):
    """Generative inpainting (LaMa, Apache-2.0) for texels no frame saw; Telea diffusion as a fallback."""
    global _lama
    try:
        if _lama is None:
            from simple_lama_inpainting import SimpleLama
            _lama = SimpleLama()
        out = np.asarray(_lama(img, hole))[:img.shape[0], :img.shape[1]]
        return np.where(hole[..., None] > 0, out, img), 'LaMa generative inpainting'
    except Exception as e:  # not installed / no GPU memory: classic inpainting still beats holes
        print(f'    (LaMa unavailable: {e.__class__.__name__}; using Telea inpainting)')
        return cv2.inpaint(img, hole, 6, cv2.INPAINT_TELEA), 'Telea inpainting'


def finish(rgb, seen, inside, fallback):
    """Fill unseen texels inside the region (generative AI); transparent outside it."""
    img = np.clip(rgb, 0, 255).astype(np.uint8)
    hole = ((inside & ~seen) * 255).astype(np.uint8)
    if seen.any():
        img[~seen] = np.median(img[seen], 0) if fallback is None else fallback  # neutral start outside the views
        img, how = generative_fill(img, hole)
        print(f'    holes filled with {how}')
    else:
        img[:] = fallback if fallback is not None else (200, 200, 200)
    alpha = (inside * 255).astype(np.uint8)
    return np.dstack([img[..., ::-1], alpha])  # BGRA for cv2.imwrite


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--splat', required=True)
    ap.add_argument('--transforms', required=True)
    ap.add_argument('--dataparser', required=True)
    ap.add_argument('--ceiling', type=float, help='ceiling height in metres (default: scene wallHeight)')
    a = ap.parse_args()

    scene_dir = Path(a.scene)
    scene = json.loads((scene_dir / 'scene.json').read_text())
    rows, cell, origin = scene['rows'], scene['cellSize'], scene['origin']
    region = dilate(np.array([[c != ' ' for c in r] for r in rows]), 2)
    cams = metric_cameras(a.transforms, a.dataparser, scene)
    ceiling = a.ceiling or scene.get('wallHeight', 2.75)

    # occlusion test: a texel is visible from a frame if nothing in the splat sits in front of it
    import depthcarve
    v = read_ply(a.splat)
    views, Kd, wd, hd = depthcarve.load_views(a.transforms, a.dataparser, max_views=10 ** 9)
    depths, alphas, Ks = depthcarve.render_depths(v, views, Kd, wd, hd, width=192)
    s_sc = scene['splat']['scale'][0]
    sx = depths.shape[2] / cams[4]

    def depth_check(i, uv, z_metric):
        px = np.clip((uv[:, 0] * sx).astype(int), 0, depths.shape[2] - 1)
        py = np.clip((uv[:, 1] * sx).astype(int), 0, depths.shape[1] - 1)
        d = depths[i, py, px] * s_sc
        empty = alphas[i, py, px] < 0.15  # truly empty splat pixel: nothing can block the view
        return empty | (z_metric < d * 1.05 + 0.15)

    # where the floor and ceiling really are: dense layers of opaque splats (exact to a few cm, unlike the
    # y=0 fit or a scale-dependent tape height)
    tr = scene['splat']
    R_sc, t_sc = Rotation.from_quat(tr['rotation']).as_matrix(), np.array(tr['position'])
    op = 1 / (1 + np.exp(-v['opacity']))
    M = s_sc * (np.stack([v['x'], v['y'], v['z']], 1)[op > 0.5] @ R_sc.T) + t_sc

    def layer(lo, hi, fallback, highest=False):
        y = M[(M[:, 1] > lo) & (M[:, 1] < hi), 1]
        if len(y) < 500:
            return fallback
        h, e = np.histogram(y, bins=int((hi - lo) / 0.02))
        if highest:  # the ceiling caps the room: topmost dense layer, not door/transom tops below it
            dense = np.flatnonzero(h > 0.3 * h.max())
            return float(e[dense[-1]] + 0.01)
        return float(e[h.argmax()] + 0.01)

    surfaces = {}
    for name, normal, y in (('floor', 1, layer(-0.3, 0.4, 0.0)), ('ceiling', -1, layer(2.0, 4.0, ceiling, highest=True))):
        print(f'  projecting frames onto the {name} (measured at y = {y:.2f} m)')
        rgb, seen, inside, step = project_plane(y, normal, region, origin, cell, cams, depth_check, print)
        cover = seen.sum() / max(1, inside.sum())
        tex = finish(rgb, seen, inside, None if cover > 0.02 else np.array([235, 235, 232]))
        cv2.imwrite(str(scene_dir / f'{name}.webp'), tex, [cv2.IMWRITE_WEBP_QUALITY, 88])
        surfaces[name] = {'url': f'{name}.webp', 'y': round(y, 3), 'x0': origin[0], 'z0': origin[1],
                          'width': round(tex.shape[1] * step, 3), 'depth': round(tex.shape[0] * step, 3),
                          'coverage': round(float(cover), 3)}
        print(f'  {name}: {cover:.0%} of the area seen by the camera, rest inpainted -> {name}.webp')
    scene['surfaces'] = surfaces
    (scene_dir / 'scene.json').write_text(json.dumps(scene, indent=1))


if __name__ == '__main__':
    main()
