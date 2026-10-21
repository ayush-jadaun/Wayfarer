"""Gaussian splat (.ply) -> floor-aligned metric occupancy grid + compact .splat + scene.json.

Steps
 1. load splats, drop transparent / oversized floaters
 2. RANSAC the floor plane, rotate so floor normal = +Y (three.js up), floor at y = 0
 3. metric scale from (priority) --scale, --ceiling + detected ceiling, camera height, default ceiling
 4. yaw-align so walls run along grid axes (maximise histogram peakiness)
 5. slice 0.12..1.9 m above floor -> obstacles; floor evidence + camera path -> walkable
 6. keep only floor connected to where the camera walked (kills floaters seen through windows/mirrors)
"""
import argparse
import json
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
PLY_TYPES = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'uchar': 'u1', 'uint8': 'u1',
             'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'short': '<i2', 'ushort': '<u2'}


# ---------------------------------------------------------------- io
def read_ply(path):
    with open(path, 'rb') as f:
        header = []
        while (line := f.readline().decode('ascii').strip()) != 'end_header':
            header.append(line)
        if not any('binary_little_endian' in l for l in header):
            raise ValueError('only binary little-endian PLY is supported')
        n = int(next(l.split()[2] for l in header if l.startswith('element vertex')))
        props = [(l.split()[2], PLY_TYPES[l.split()[1]]) for l in header if l.startswith('property')]
        return np.fromfile(f, dtype=np.dtype(props), count=n)


def write_splat(v, path, min_alpha=0.02):
    """antimatter15 .splat: 32 B/splat (pos f32x3, scale f32x3, rgba u8x4, rot u8x4). ~8x smaller than PLY."""
    alpha = sigmoid(v['opacity'])
    keep = alpha >= min_alpha
    v, alpha = v[keep], alpha[keep]
    scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1))
    order = np.argsort(-scales.prod(1) * alpha)  # big opaque splats first -> nicer progressive load
    rgb = 0.5 + SH_C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1)
    rot = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    out = np.empty(len(v), dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
    out['p'] = np.stack([v['x'], v['y'], v['z']], 1)
    out['s'] = scales
    out['c'][:, :3] = np.clip(rgb * 255, 0, 255)
    out['c'][:, 3] = np.clip(alpha * 255, 0, 255)
    out['r'] = np.clip(rot * 128 + 128, 0, 255)
    out[order].tofile(path)
    return len(v)


def load_cameras(transforms_json, dataparser_json):
    """Camera centres + forward vectors in the splat's (nerfstudio dataparser) frame."""
    # capture order (frame names are sequential); COLMAP's own order is by registration, not time
    frames = sorted(json.loads(Path(transforms_json).read_text())['frames'], key=lambda f: f['file_path'])
    dp = json.loads(Path(dataparser_json).read_text())
    T, s = np.array(dp['transform']), dp['scale']
    c2w = np.array([f['transform_matrix'] for f in frames])[:, :3, :]
    pos = s * (c2w[:, :, 3] @ T[:, :3].T + T[:, 3])
    fwd = -(c2w[:, :, 2] @ T[:, :3].T)  # OpenGL camera looks down -z
    return pos, fwd


# ---------------------------------------------------------------- geometry
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ransac_plane(P, up, rng, max_tilt=25, iters=500, tol=0.01):
    """Largest plane whose normal is within max_tilt degrees of `up`. Returns (normal toward up, point, inlier mask)."""
    best_n, best_c, best_cnt = None, None, -1
    cos_tilt = math.cos(math.radians(max_tilt))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if (norm := np.linalg.norm(n)) < 1e-12:
            continue
        n /= norm
        if n @ up < 0:
            n = -n
        if n @ up < cos_tilt:
            continue
        cnt = np.count_nonzero(np.abs((P - a) @ n) < tol)
        if cnt > best_cnt:
            best_n, best_c, best_cnt = n, a, cnt
    if best_n is None:
        raise RuntimeError('no horizontal plane found; check --up')
    inl = np.abs((P - best_c) @ best_n) < tol
    Q = P[inl]
    c = Q.mean(0)
    n = np.linalg.eigh(np.cov((Q - c).T))[1][:, 0]  # least-squares refine: smallest-variance axis
    return (n if n @ up > 0 else -n), c, inl


def basis_from_up(n):
    """Rotation matrix (rows = new x, y, z axes) taking `n` to +Y."""
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
    x = ref - (ref @ n) * n
    x /= np.linalg.norm(x)
    z = np.cross(x, n)
    return np.stack([x, n, z])


def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def manhattan_yaw(xz, bin_size=0.05):
    """Yaw (rad) that makes walls axis-aligned: maximise sum of squared histogram mass along x and z."""
    best, best_score = 0.0, -1
    for deg in np.arange(0, 90, 1.0):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        x, z = c * xz[:, 0] + s * xz[:, 1], -s * xz[:, 0] + c * xz[:, 1]
        score = 0
        for a in (x, z):
            h = np.bincount(((a - a.min()) / bin_size).astype(int)).astype(float)
            score += (h / h.sum()) @ (h / h.sum())
        if score > best_score:
            best, best_score = t, score
    return best


def mat_to_quat(R):
    """Rotation matrix -> [x, y, z, w]."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [x, y, z, w]


# ---------------------------------------------------------------- grid morphology (numpy only)
def dilate(m, r=1, outside=False):
    p = np.pad(m, r, constant_values=outside)
    out = np.zeros_like(m)
    for dz in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dz:dz + m.shape[0], dx:dx + m.shape[1]]
    return out


def erode(m, r=1):
    return ~dilate(~m, r, outside=True)  # off-grid counts as empty, so erosion always terminates


def components_touching(free, seeds):
    """4-connected flood fill of `free` from seed cells."""
    keep = np.zeros_like(free)
    q = deque((z, x) for z, x in seeds if free[z, x])
    for z, x in q:
        keep[z, x] = True
    while q:
        z, x = q.popleft()
        for nz, nx in ((z + 1, x), (z - 1, x), (z, x + 1), (z, x - 1)):
            if 0 <= nz < free.shape[0] and 0 <= nx < free.shape[1] and free[nz, nx] and not keep[nz, nx]:
                keep[nz, nx] = True
                q.append((nz, nx))
    return keep


# ---------------------------------------------------------------- main build
def build(v, cameras=None, *, up=(0, 0, 1), scale=None, marker=None, ceiling=None, camera_height=1.4, default_ceiling=2.75,
          cell=0.1, obstacle_band=(0.12, 1.9), tall_from=1.2, seed=0, log=print):
    rng = np.random.default_rng(seed)
    P = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    alpha = sigmoid(v['opacity']) if 'opacity' in v.dtype.names else np.ones(len(P))
    keep = alpha > 0.3
    if 'scale_0' in v.dtype.names:
        size = np.exp(np.max(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1), 1))
        keep &= size <= np.quantile(size, 0.98)
    P, alpha = P[keep], alpha[keep]
    log(f'  {len(P):,} solid splats after floater filtering')

    up = np.asarray(up, float) / np.linalg.norm(up)
    extent = np.linalg.norm(np.quantile(P, 0.98, 0) - np.quantile(P, 0.02, 0))
    tol = extent * 0.004
    h = P @ up
    sub = lambda m: P[m][rng.choice(np.count_nonzero(m), min(40000, np.count_nonzero(m)), replace=False)]

    n, c_floor, _ = ransac_plane(sub(h < np.quantile(h, 0.35)), up, rng, tol=tol)
    R = basis_from_up(n)
    hh = (P - c_floor) @ n  # raw height above floor

    cam_raw = cam_top = None
    if cameras is not None:
        cam_h = (cameras[0] - c_floor) @ n
        cam_raw, cam_top = float(np.median(cam_h)), float(np.quantile(cam_h, 0.95))

    # the ceiling must sit above every camera: rejects floater layers that form around the capture height
    ceil_raw = None
    top = hh > (cam_top * 1.15 if cam_top and cam_top > 0 else np.quantile(hh, 0.6))
    if np.count_nonzero(top) > 200:
        try:
            nc, cc, inl = ransac_plane(sub(top), n, rng, max_tilt=8, tol=tol)
            cand = float((cc - c_floor) @ n)
            above = np.count_nonzero(hh[top] > cand + 3 * tol) / np.count_nonzero(top)
            if inl.mean() > 0.08 and above < 0.05:  # a ceiling caps the room: almost nothing above it
                ceil_raw = cand
        except RuntimeError:
            pass

    estimates = {}
    if marker:
        estimates['marker'] = marker
    if ceil_raw:
        estimates['ceiling'] = (ceiling or default_ceiling) / ceil_raw
    if cam_raw and cam_raw > 0:
        estimates['camera'] = camera_height / cam_raw
    if scale:
        s, src = scale, '--scale'
    elif marker:
        s, src = marker, 'ArUco marker'
    elif ceiling and 'ceiling' in estimates:
        s, src = estimates['ceiling'], f'ceiling {ceiling} m'
    elif 'camera' in estimates:
        s, src = estimates['camera'], f'camera height {camera_height} m'
    elif 'ceiling' in estimates:
        s, src = estimates['ceiling'], f'assumed ceiling {default_ceiling} m'
    else:
        raise RuntimeError('cannot infer metric scale: pass --scale, or capture with the ceiling visible')
    log(f'  scale {s:.4f} m/unit from {src}  ' + '  '.join(f'[{k}: {e:.4f}]' for k, e in estimates.items()))
    for k, e in estimates.items():
        if abs(e / s - 1) > 0.12:
            log(f'  WARNING: {k} estimate disagrees with the chosen scale by {abs(e / s - 1):.0%}; '
                'check measurements or pass --scale')

    # aligned metric coordinates (before yaw): y up, floor at 0
    A = s * ((P - c_floor) @ R.T)
    lo, hi = obstacle_band
    if ceil_raw:
        hi = min(hi, ceil_raw * s - 0.25)  # never treat the ceiling as an obstacle
    wall = (A[:, 1] > lo) & (A[:, 1] < hi)
    yaw = manhattan_yaw(A[wall][rng.choice(np.count_nonzero(wall), min(60000, np.count_nonzero(wall)), replace=False)][:, [0, 2]])
    Ry = rot_y(yaw)  # same rotation manhattan_yaw scored: walls land on the axes
    R_total = Ry @ R
    A = A @ Ry.T
    t = -s * (R_total @ c_floor)  # p' = s R p + t
    log(f'  floor tilt {math.degrees(math.acos(min(1, n @ up))):.1f} deg vs up hint, wall yaw {math.degrees(yaw):.1f} deg')

    # grid extent from solid geometry near the floor, clamped around where the camera actually went
    near = (A[:, 1] > -0.1) & (A[:, 1] < hi)
    x0, z0 = np.quantile(A[near][:, [0, 2]], 0.003, 0) - 0.3
    x1, z1 = np.quantile(A[near][:, [0, 2]], 0.997, 0) + 0.3
    if cameras is not None:
        C = s * (cameras[0] @ R_total.T) + t
        (cx0, cz0), (cx1, cz1) = C[:, [0, 2]].min(0) - 4, C[:, [0, 2]].max(0) + 4
        x0, z0, x1, z1 = max(x0, cx0), max(z0, cz0), min(x1, cx1), min(z1, cz1)
    W, H = int(math.ceil((x1 - x0) / cell)), int(math.ceil((z1 - z0) / cell))
    ix = np.clip(((A[:, 0] - x0) / cell).astype(int), 0, W - 1)
    iz = np.clip(((A[:, 2] - z0) / cell).astype(int), 0, H - 1)
    flat = iz * W + ix
    count = lambda m: np.bincount(flat[m], weights=alpha[m], minlength=W * H).reshape(H, W)

    obs_n, tall_n = count(wall), count((A[:, 1] > tall_from) & (A[:, 1] < hi))
    near_floor = np.abs(A[:, 1]) < 0.15
    band = float(np.clip(3 * A[near_floor, 1].std(), 0.04, 0.12)) if near_floor.any() else 0.06  # floor 'thickness'
    floor_n = count(np.abs(A[:, 1]) < band)
    nz = obs_n[obs_n > 0]
    thr = max(2.0, 0.025 * np.quantile(nz, 0.9)) if len(nz) else 2.0  # walls are dense, table tops are not
    obstacle = drop_small_blobs(obs_n >= thr, min_cells=max(2, round(0.03 / cell ** 2)))  # < 0.03 m^2 = floater
    tall = obstacle & (tall_n >= thr * 0.5)
    fz = floor_n[floor_n > 0]
    floor = floor_n >= (0.2 * np.median(fz) if len(fz) else 1)  # real floor is dense; lone floaters are not

    cam_cells, path = [], np.zeros((H, W), bool)
    if cameras is not None:
        cxs, czs = ((C[:, 0] - x0) / cell).astype(int), ((C[:, 2] - z0) / cell).astype(int)
        for i, (cx, cz) in enumerate(zip(cxs, czs)):
            if 0 <= cx < W and 0 <= cz < H:
                cam_cells.append((cz, cx))
            # the walk between consecutive frames is passable: a doorway you walked through can never be sealed
            if i and max(abs(cx - cxs[i - 1]), abs(cz - czs[i - 1])) <= round(1.5 / cell):  # skip pose jumps
                n = max(abs(cx - cxs[i - 1]), abs(cz - czs[i - 1])) + 1
                lx = np.clip(np.round(np.linspace(cxs[i - 1], cx, n)).astype(int), 0, W - 1)
                lz = np.clip(np.round(np.linspace(czs[i - 1], cz, n)).astype(int), 0, H - 1)
                path[lz, lx] = True
        for cz, cx in cam_cells:
            path[cz, cx] = True
        path = dilate(path, 1)
        floor |= path
        obstacle &= ~path
        tall &= ~path

    floor = erode(dilate(floor, 2), 2)  # close holes in textureless floor
    # free space = no obstacle and near floor evidence; walls stop the flood fill, so sparse floors still fill in.
    # Seal wall gaps < 40 cm for the fill (doors are >= 70 cm), then grow back up to the real wall edge.
    seal = max(1, round(0.2 / cell))
    free = ~obstacle & dilate(floor, 5)
    sealed = (free & ~dilate(obstacle, seal)) | path
    seeds = [c for c in cam_cells if sealed[c]] or cam_cells
    if seeds:
        walk = components_touching(sealed, seeds)
    else:  # largest component
        walk = components_touching(sealed, [tuple(np.argwhere(sealed)[np.argmax(dist_from_blocked(sealed)[sealed])])])
    for _ in range(seal):
        walk = dilate(walk, 1) & free
    log(f'  grid {W}x{H} @ {cell} m, {walk.sum() * cell * cell:.1f} m^2 walkable, obstacle threshold {thr:.1f}, floor band {band:.2f} m')

    grid = np.full((H, W), ' ')
    grid[walk] = '.'
    # keep whole obstacle blobs that touch reachable floor; drop far-away floaters
    near = components_touching(obstacle, [tuple(c) for c in np.argwhere(obstacle & dilate(walk, 1))])
    grid[near] = 'o'
    grid[near & tall] = '#'

    if cameras is not None and cam_cells:
        # start where the capture started, facing into the open space (the first frame often faces a wall)
        C = s * (cameras[0] @ R_total.T) + t
        wz, wx = np.argwhere(walk).mean(0)
        mx, mz = x0 + (wx + 0.5) * cell, z0 + (wz + 0.5) * cell
        spawn = {'x': round(float(C[0, 0]), 2), 'z': round(float(C[0, 2]), 2),
                 'yaw': round(math.atan2(-(mx - C[0, 0]), -(mz - C[0, 2])), 3)}
    else:
        d = dist_from_blocked(walk)
        cz, cx = np.unravel_index(np.argmax(d), d.shape)
        spawn = {'x': round(x0 + (cx + 0.5) * cell, 2), 'z': round(z0 + (cz + 0.5) * cell, 2), 'yaw': 0}

    return {
        'cellSize': cell, 'origin': [round(float(x0), 3), round(float(z0), 3)], 'floorY': 0,
        'wallHeight': round(ceil_raw * s, 2) if ceil_raw else default_ceiling, 'eyeHeight': 1.6,
        'spawn': spawn,
        'transform': {'position': [float(x) for x in t], 'rotation': mat_to_quat(R_total), 'scale': [float(s)] * 3},
        'rows': [''.join(r) for r in grid],
    }


def drop_small_blobs(mask, min_cells):
    """Remove 8-connected blobs smaller than min_cells (isolated floaters), keep everything else."""
    out, seen = mask.copy(), np.zeros_like(mask)
    H, W = mask.shape
    for z, x in np.argwhere(mask):
        if seen[z, x]:
            continue
        blob, q = [], deque([(z, x)])
        seen[z, x] = True
        while q:
            cz, cx = q.popleft()
            blob.append((cz, cx))
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nz, nx = cz + dz, cx + dx
                    if 0 <= nz < H and 0 <= nx < W and mask[nz, nx] and not seen[nz, nx]:
                        seen[nz, nx] = True
                        q.append((nz, nx))
        if len(blob) < min_cells:
            for cz, cx in blob:
                out[cz, cx] = False
    return out


def to_sog(ply, out):
    """Compress with PlayCanvas splat-transform (installed in web/): ~10 B/splat with full SH, floaters filtered."""
    cli = Path(__file__).resolve().parent.parent / 'web' / 'node_modules' / '@playcanvas' / 'splat-transform' / 'bin' / 'cli.mjs'
    if not cli.exists() or not shutil.which('node'):
        return False
    r = subprocess.run(['node', str(cli), '-w', '--no-tty', '-q', str(ply), '-N', '-F', str(out)])
    return r.returncode == 0 and out.exists()


def dist_from_blocked(free):
    """Chebyshev distance to nearest non-free cell (cheap iterative erosion)."""
    d, m, k = np.zeros(free.shape, int), free.copy(), 0
    while m.any():
        k += 1
        d[m] = k
        m = erode(m, 1)
    return d


def save_preview(rows, path):
    try:
        from PIL import Image
    except ImportError:
        return
    pal = {'.': (24, 44, 70), 'o': (76, 195, 230), '#': (240, 240, 240), ' ': (8, 14, 24)}
    img = Image.new('RGB', (len(rows[0]), len(rows)))
    img.putdata([pal[c] for r in rows for c in r])
    img.resize((img.width * 4, img.height * 4), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('splat', help='trained Gaussian splat .ply (nerfstudio ns-export gaussian-splat)')
    ap.add_argument('--out', required=True, help='scene folder, e.g. web/public/scenes/home')
    ap.add_argument('--name', default=None, help='display name')
    ap.add_argument('--transforms', help='transforms.json from ns-process-data (camera path)')
    ap.add_argument('--dataparser', help='dataparser_transforms.json from the training run')
    ap.add_argument('--up', default='0,0,1', help='rough up axis of the splat (nerfstudio: 0,0,1)')
    ap.add_argument('--scale', type=float, help='metres per splat unit (overrides everything)')
    ap.add_argument('--ceiling', type=float, help='measured floor-to-ceiling height in metres')
    ap.add_argument('--camera-height', type=float, default=1.4, help='SOP phone height in metres')
    ap.add_argument('--cell', type=float, default=0.1, help='grid cell size in metres')
    ap.add_argument('--marker-size', type=float, default=0.15, help='printed ArUco marker side in metres')
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'reading {a.splat}')
    v = read_ply(a.splat)
    cams, marker = None, None
    if a.transforms and a.dataparser:
        cams = load_cameras(a.transforms, a.dataparser)
        if not a.scale:
            from marker_scale import marker_scale
            marker = marker_scale(a.transforms, a.dataparser, a.marker_size)
    scene = build(v, cams, up=[float(x) for x in a.up.split(',')], scale=a.scale, marker=marker, ceiling=a.ceiling,
                  camera_height=a.camera_height, cell=a.cell)

    for stale in ('scene.sog', 'scene.splat', 'scene.ply'):
        (out / stale).unlink(missing_ok=True)
    if 'f_dc_0' in v.dtype.names and to_sog(a.splat, out / 'scene.sog'):
        url = 'scene.sog'
        print(f'  wrote scene.sog ({(out / url).stat().st_size / 1e6:.1f} MB, full SH)')
    elif 'f_dc_0' in v.dtype.names:  # node/splat-transform unavailable: our own 32 B/splat packing
        n = write_splat(v, out / 'scene.splat')
        url = 'scene.splat'
        print(f'  wrote scene.splat ({n:,} splats, {(out / url).stat().st_size / 1e6:.1f} MB)')
    else:
        shutil.copy(a.splat, out / 'scene.ply')
        url = 'scene.ply'

    old = out / 'scene.json'
    pois = json.loads(old.read_text()).get('pois', []) if old.exists() else []  # re-runs keep tagged POIs
    tr = scene.pop('transform')
    scene = {'name': a.name or out.name, **scene, 'splat': {'url': url, 'count': int(len(v)), **tr}, 'pois': pois}
    old.write_text(json.dumps(scene, indent=1))
    save_preview(scene['rows'], out / 'grid.png')
    print(f'done -> {out / "scene.json"}')


if __name__ == '__main__':
    main()
