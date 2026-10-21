"""Auto-label the twin: detect furniture/appliances in the training frames and place them as POIs.

COCO object detector (torchvision Faster R-CNN, BSD) on the frames -> 3D position of each box from the
trained splat's rendered depth -> cluster across views -> keep objects seen from >= 3 frames.

  python pipeline/autopoi.py --scene web/public/scenes/home --splat work/home/export/splat.ply \
      --transforms work/home/colmap/transforms.json --dataparser <run>/dataparser_transforms.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

# COCO class -> POI name (what someone would search for indoors)
LABELS = {'bed': 'Bed', 'couch': 'Sofa', 'chair': 'Chair', 'tv': 'TV', 'refrigerator': 'Fridge', 'oven': 'Oven',
          'microwave': 'Microwave', 'sink': 'Sink', 'toilet': 'Toilet', 'dining table': 'Table', 'potted plant': 'Plant',
          'laptop': 'Laptop', 'clock': 'Clock', 'bench': 'Bench', 'toaster': 'Toaster', 'suitcase': 'Suitcase'}
# merge radius ~ half the object's size: one 2 m bed seen from both ends is still one bed
RADIUS = {'bed': 2.0, 'couch': 1.6, 'dining table': 1.0, 'bench': 1.0, 'refrigerator': 0.8}
# floor furniture cannot float: centre height must be below this (metres)
MAX_HEIGHT = {'bed': 1.0, 'couch': 1.0, 'chair': 1.1, 'dining table': 1.1, 'bench': 0.9, 'toilet': 0.9,
              'refrigerator': 1.6, 'potted plant': 1.6, 'suitcase': 1.0}


def detect(paths, min_score):
    import torch
    from PIL import Image
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=min_score).eval().cuda()
    names = weights.meta['categories']
    tfm = weights.transforms()
    out = []
    with torch.no_grad():
        for p in paths:
            img = tfm(Image.open(p).convert('RGB')).cuda()
            r = model([img])[0]
            dets = [(names[l], float(s), b.tolist()) for l, s, b in zip(r['labels'], r['scores'], r['boxes'])
                    if names[l] in LABELS]
            out.append(dets)
    return out


def cluster(points, radius):
    """Greedy clustering on xz; returns list of index arrays."""
    left, groups = list(range(len(points))), []
    while left:
        seed = left.pop(0)
        members = [seed]
        centre = points[seed]
        for _ in range(3):  # re-centre a few times
            members = [i for i in [seed] + left if np.linalg.norm(points[i][[0, 2]] - centre[[0, 2]]) < radius]
            centre = np.median(points[members], 0)
        left = [i for i in left if i not in members]
        groups.append(np.array(members))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help='scene folder with scene.json')
    ap.add_argument('--splat', required=True)
    ap.add_argument('--transforms', required=True)
    ap.add_argument('--dataparser', required=True)
    ap.add_argument('--min-score', type=float, default=0.6)
    ap.add_argument('--min-views', type=int, default=3)
    ap.add_argument('--every', type=int, default=2, help='use every n-th frame')
    a = ap.parse_args()

    import depthcarve
    from navgrid import read_ply
    scene_file = Path(a.scene) / 'scene.json'
    scene = json.loads(scene_file.read_text())
    tf = json.loads(Path(a.transforms).read_text())
    frames = sorted(tf['frames'], key=lambda f: f['file_path'])[::a.every]
    root = Path(a.transforms).parent
    views, K, w, h = depthcarve.load_views(a.transforms, a.dataparser, max_views=10 ** 9)
    views = views[::a.every]
    print(f'detecting objects in {len(frames)} frames')
    dets = detect([root / f['file_path'] for f in frames], a.min_score)
    n_det = sum(len(d) for d in dets)
    print(f'  {n_det} detections of {len({x[0] for d in dets for x in d})} kinds')
    if not n_det:
        return

    v = read_ply(a.splat)
    width = 320
    depths, alphas, Ks = depthcarve.render_depths(v, views, K, w, h, width=width)
    sc = width / w
    tr = scene['splat']
    from scipy.spatial.transform import Rotation  # noqa: E402  (scipy ships with nerfstudio)
    R = Rotation.from_quat(tr['rotation']).as_matrix()
    s, t = tr['scale'][0], np.array(tr['position'])
    hits = {}  # label -> list of (metric point, score, frame)
    for fi, frame_dets in enumerate(dets):
        for label, score, (x0, y0, x1, y1) in frame_dets:
            # median depth over the box centre (inner 40%), where the object itself is most likely
            cx0, cx1 = int((x0 + 0.3 * (x1 - x0)) * sc), int((x1 - 0.3 * (x1 - x0)) * sc) + 1
            cy0, cy1 = int((y0 + 0.3 * (y1 - y0)) * sc), int((y1 - 0.3 * (y1 - y0)) * sc) + 1
            d, al = depths[fi, cy0:cy1, cx0:cx1], alphas[fi, cy0:cy1, cx0:cx1]
            d = d[(al > 0.6) & (d > 0)]
            if len(d) < 3:
                continue
            u, vv = (x0 + x1) / 2 * sc, (y0 + y1) / 2 * sc
            ray = np.array([(u - Ks[0, 2]) / Ks[0, 0], (vv - Ks[1, 2]) / Ks[1, 1], 1.0])
            p = views[fi][:3, :3] @ (ray * np.median(d)) + views[fi][:3, 3]
            m = s * (R @ p) + t
            if -0.2 < m[1] < MAX_HEIGHT.get(label, 3.2):  # reject depth outliers (box centre hit the wall behind)
                hits.setdefault(label, []).append((m, score, fi))

    pois = [p for p in scene.get('pois', []) if not str(p.get('id', '')).startswith('auto-')]  # re-runs replace
    added = []
    for label, hl in hits.items():
        pts = np.array([x[0] for x in hl])
        for g in cluster(pts, radius=RADIUS.get(label, 0.6)):
            frames_seen = {hl[i][2] for i in g}
            if len(frames_seen) < a.min_views:
                continue
            c = np.median(pts[g], 0)
            conf = float(np.mean([hl[i][1] for i in g]))
            added.append({'name': LABELS[label], 'category': 'item', 'x': round(float(c[0]), 2), 'z': round(float(c[2]), 2),
                          'y': round(float(min(max(c[1], 0.4), 2.0)), 2),
                          'desc': f'auto-detected · seen in {len(frames_seen)} views · confidence {conf:.0%}', 'views': len(frames_seen)})
    # number duplicates ("Chair 2") most-seen first
    added.sort(key=lambda p: -p['views'])
    count = {}
    for p in added:
        count[p['name']] = count.get(p['name'], 0) + 1
        if count[p['name']] > 1:
            p['name'] = f"{p['name']} {count[p['name']]}"
        p['id'] = 'auto-' + p['name'].lower().replace(' ', '-')
        del p['views']
    scene['pois'] = pois + added
    scene_file.write_text(json.dumps(scene, indent=1))
    print(f'  placed {len(added)} POIs: ' + ', '.join(p['name'] for p in added))


if __name__ == '__main__':
    main()
