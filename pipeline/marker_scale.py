"""Metric scale from a printed ArUco marker filmed during capture (docs/marker.svg).

Detect the marker in the training frames, triangulate its 4 corners from the known camera poses, and
compare the reconstructed side length with the printed one. Accurate to ~1% when the marker is seen
from 3+ angles, versus ~10% for the camera-height prior.
"""
import json
from pathlib import Path

import numpy as np

DICT = 'DICT_4X4_50'


def detect(images, K, dist, min_side_px=24):
    """{marker_id: [(frame_idx, 4x2 undistorted normalised corners)]}"""
    import cv2
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT)),
                                  cv2.aruco.DetectorParameters())
    seen = {}
    for i, path in enumerate(images):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        corners, ids, _ = det.detectMarkers(img)
        for c, mid in zip(corners, ids.ravel() if ids is not None else []):
            c = c.reshape(4, 2)
            if np.linalg.norm(c[0] - c[2]) < min_side_px * 1.4:  # too small to trust the corners
                continue
            norm = cv2.undistortPoints(c.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(4, 2)
            seen.setdefault(int(mid), []).append((i, norm))
    return seen


def triangulate(w2c, obs):
    """Linear (DLT) triangulation of one point from normalised image coords in several views."""
    A = []
    for P, (x, y) in zip(w2c, obs):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def marker_scale(transforms_json, dataparser_json, marker_size, log=print):
    """Metres per splat unit, or None when no marker is seen from enough views."""
    tf = json.loads(Path(transforms_json).read_text())
    root = Path(transforms_json).parent
    K = np.array([[tf['fl_x'], 0, tf['cx']], [0, tf['fl_y'], tf['cy']], [0, 0, 1]])
    dist = np.array([tf.get(k, 0.0) for k in ('k1', 'k2', 'p1', 'p2')])
    frames = tf['frames']
    # nerfstudio stores OpenGL camera-to-world; OpenCV projection needs world-to-camera with y/z flipped
    w2c = [np.linalg.inv(np.array(f['transform_matrix']) @ np.diag([1, -1, -1, 1]))[:3] for f in frames]
    seen = detect([root / f['file_path'] for f in frames], K, dist)

    best = None
    for mid, obs in seen.items():
        if len(obs) < 3:
            continue
        idx = [i for i, _ in obs]
        corners = np.array([triangulate([w2c[i] for i in idx], [o[k] for _, o in obs]) for k in range(4)])
        sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        spread = sides.std() / sides.mean()
        if spread > 0.05:  # not a square: bad detections or bad poses
            log(f'  marker {mid}: rejected, sides disagree by {spread:.1%}')
            continue
        if best is None or len(obs) > best[0]:
            best = (len(obs), mid, sides.mean(), spread)
    if not best:
        if seen:
            log('  marker seen but not from 3+ usable views; ignoring it')
        return None
    n, mid, side_tf, spread = best
    dp_scale = json.loads(Path(dataparser_json).read_text())['scale']  # transforms frame -> splat frame
    s = marker_size / (side_tf * dp_scale)
    log(f'  ArUco marker {mid}: {n} views, sides consistent to {spread:.1%} -> {s:.4f} m/unit')
    return s


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('transforms')
    ap.add_argument('dataparser')
    ap.add_argument('--marker-size', type=float, default=0.15)
    a = ap.parse_args()
    print(marker_scale(a.transforms, a.dataparser, a.marker_size))
