"""Self-check: render an ArUco marker from known poses -> marker_scale must recover the metric scale.
Run: .venv-ns/Scripts/python pipeline/test_marker_scale.py   (needs opencv, which nerfstudio installs)
"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from marker_scale import marker_scale

W, H, F = 960, 720, 800.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
SIDE_TF = 0.3      # marker side in transforms.json units
DP_SCALE = 0.5     # dataparser scale: splat units = transforms units * 0.5
PRINTED = 0.15     # metres
EXPECTED = PRINTED / (SIDE_TF * DP_SCALE)  # = 1.0 m per splat unit

marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 200)
pad = 40
tex = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
full = SIDE_TF * tex.shape[0] / 200  # plane size including the white border
# marker plane: z = 0, centred at origin, texture x -> +x, texture y -> -y
plane = np.array([[-full / 2, full / 2, 0], [full / 2, full / 2, 0], [full / 2, -full / 2, 0], [-full / 2, -full / 2, 0]])
src = np.float32([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]])


def look_at(eye, target):
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / 'images').mkdir()
    frames = []
    for i, ang in enumerate(np.radians([-35, -15, 0, 20, 40])):
        eye = np.array([0.9 * np.sin(ang), 0.15 * (i - 2), 0.9 * np.cos(ang)])
        c2w = look_at(eye, np.zeros(3))
        w2c = np.linalg.inv(c2w)
        cam = (w2c[:3, :3] @ plane.T).T + w2c[:3, 3]
        px = (K @ cam.T).T
        dst = np.float32(px[:, :2] / px[:, 2:])
        img = np.full((H, W), 128, np.uint8)
        cv2.warpPerspective(tex, cv2.getPerspectiveTransform(src, dst), (W, H), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
        cv2.imwrite(str(d / 'images' / f'f{i}.png'), img)
        frames.append({'file_path': f'images/f{i}.png', 'transform_matrix': (c2w @ np.diag([1, -1, -1, 1])).tolist()})
    json.dump({'fl_x': F, 'fl_y': F, 'cx': W / 2, 'cy': H / 2, 'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0, 'frames': frames},
              open(d / 'transforms.json', 'w'))
    json.dump({'transform': np.eye(4)[:3].tolist(), 'scale': DP_SCALE}, open(d / 'dp.json', 'w'))

    s = marker_scale(d / 'transforms.json', d / 'dp.json', PRINTED, log=lambda *_: None)
    assert s is not None, 'marker not found'
    assert abs(s / EXPECTED - 1) < 0.01, f'scale {s:.4f}, expected {EXPECTED:.4f}'
    # fewer than 3 views -> no estimate (two views can hide pose errors)
    t = json.loads((d / 'transforms.json').read_text())
    t['frames'] = t['frames'][:2]
    (d / 'two.json').write_text(json.dumps(t))
    assert marker_scale(d / 'two.json', d / 'dp.json', PRINTED, log=lambda *_: None) is None
print(f'ok: marker scale {s:.4f} m/unit (expected {EXPECTED:.4f})')
