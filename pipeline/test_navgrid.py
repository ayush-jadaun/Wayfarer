"""Self-check: synthetic splat room in a tilted/rotated/scaled frame -> navgrid must recover metric layout.
Run: python pipeline/test_navgrid.py
"""
import math
import os
import tempfile

import numpy as np

from navgrid import build, write_splat

rng = np.random.default_rng(1)


def box_surface(x0, x1, y0, y1, z0, z1, n):
    """n points on the vertical faces + top of an axis-aligned box (metric, y up)."""
    p = rng.uniform([x0, y0, z0], [x1, y1, z1], (n, 3))
    face = rng.integers(0, 5, n)
    p[face == 0, 0] = x0; p[face == 1, 0] = x1
    p[face == 2, 2] = z0; p[face == 3, 2] = z1
    p[face == 4, 1] = y1
    return p


# 6 x 4 m flat, 2.6 m ceiling, partition at x=3 with a door at z 1.5..2.5, a 1x1 m table in room A
W, D, H = 6.0, 4.0, 2.6
pts = [
    rng.uniform([0, 0, 0], [W, 0, D], (40000, 3)),           # floor
    rng.uniform([0, H, 0], [W, H, D], (30000, 3)),           # ceiling
    box_surface(0, W, 0, H, 0, D, 60000),                    # outer walls (top face is ~ceiling)
    box_surface(2.95, 3.05, 0, H, 0, 1.5, 8000),             # partition, part 1
    box_surface(2.95, 3.05, 0, H, 2.5, D, 8000),             # partition, part 2
    box_surface(0.8, 1.8, 0, 0.75, 0.8, 1.8, 6000),          # table
]
P = np.concatenate(pts)
cam_m = np.stack([np.linspace(0.6, 5.4, 60), np.full(60, 1.4), np.full(60, 2.0)], 1)  # walk through the door

# raw frame: y-up -> z-up, 15 deg yaw, 2 deg tilt, 1 unit = 3 m, offset
TRUE_SCALE = 3.0
yaw, tilt = math.radians(15), math.radians(2)
to_zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
Rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
M = Rx @ Rz @ to_zup
raw = lambda p: (p @ M.T) / TRUE_SCALE + np.array([0.3, -0.2, 0.5])

R = raw(P)
v = np.zeros(len(R), dtype=[(k, '<f4') for k in ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2']])
v['x'], v['y'], v['z'] = R.T
v['opacity'] = 4.0
v['scale_0'] = v['scale_1'] = v['scale_2'] = math.log(0.003)
cams = (raw(cam_m), np.tile(np.array([1.0, 0, 0]) @ M.T, (60, 1)))

scene = build(v, cams, ceiling=H, log=lambda *_: None)
rows, cell = scene['rows'], scene['cellSize']
s = scene['transform']['scale'][0]
flat = ''.join(rows)

assert abs(s - TRUE_SCALE) / TRUE_SCALE < 0.03, f'scale {s}'
assert abs(scene['wallHeight'] - H) < 0.1, scene['wallHeight']
walk_m2 = flat.count('.') * cell * cell
assert 18 < walk_m2 < 24, f'walkable {walk_m2:.1f} m^2 (room 24 minus walls/table)'
table_m2 = flat.count('o') * cell * cell
assert 0.6 < table_m2 < 1.8, f'table footprint {table_m2:.2f} m^2'
# walls are axis aligned after yaw recovery: some column is almost entirely '#' (outer wall)
col_wall = max(max(sum(r[x] == '#' for r in rows) for x in range(len(rows[0]))), max(r.count('#') for r in rows))
assert col_wall > 0.8 * D / cell, f'longest wall column {col_wall} cells'
# both rooms reachable through the door = a single walkable region spanning > 5 m in x
# (yaw alignment may land the long side on either axis)
cells = [(x, z) for z, r in enumerate(rows) for x, c in enumerate(r) if c == '.']
span = max(max(c[i] for c in cells) - min(c[i] for c in cells) for i in (0, 1))
assert span * cell > 5, 'door not walkable'
# spawn = first camera, facing along the walk (toward the middle of the flat)
sp, (ox, oz) = scene['spawn'], scene['origin']
cx, cz = (ox + (np.mean([c[0] for c in cells]) + 0.5) * cell, oz + (np.mean([c[1] for c in cells]) + 0.5) * cell)
fwd, to_mid = np.array([-math.sin(sp['yaw']), -math.cos(sp['yaw'])]), np.array([cx - sp['x'], cz - sp['z']])
assert fwd @ to_mid / np.linalg.norm(to_mid) > 0.95, (sp, cx, cz)

print('\n'.join(rows[::2]))
print(f'ok: scale {s:.3f}, walkable {walk_m2:.1f} m^2, table {table_m2:.2f} m^2, wall {scene["wallHeight"]} m')

# compact .splat export: 32 bytes per splat, transparent ones pruned
sv = np.zeros(3, dtype=[(k, '<f4') for k in ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
                                             'f_dc_0', 'f_dc_1', 'f_dc_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']])
sv['opacity'] = [5, 5, -10]  # last one is ~transparent
sv['rot_0'] = 1
with tempfile.TemporaryDirectory() as d:
    assert write_splat(sv, os.path.join(d, 's.splat')) == 2
    assert os.path.getsize(os.path.join(d, 's.splat')) == 64
print('ok: .splat writer')
