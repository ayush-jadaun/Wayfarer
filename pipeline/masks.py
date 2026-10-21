"""Remove people (and pets) from the reconstruction, Magic-Eraser style but correct for 3D.

A photo eraser hallucinates background behind a person. For a twin we can do better: mask the person
out so COLMAP ignores features on them and splat training ignores their pixels. Other frames that saw
the same spot without the person then supply the *real* background. No ghosts, no floaters.

Masks: white = keep, black = ignore (both COLMAP and nerfstudio use this convention).

Opt-in (--mask-people): a single-image detector can't tell a visitor from a poster or a pile of clothes,
and masking a *static* object in every frame would erase it from the twin. Use it for busy venues.
"""
from pathlib import Path

import numpy as np
from PIL import Image

TRANSIENT = {'person', 'dog', 'cat'}


def detect(paths, min_score=0.85, min_area=0.004, pad=0.08):
    """Boolean keep-masks per image (False on transient objects)."""
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=min_score).eval().cuda()
    names, tfm = weights.meta['categories'], weights.transforms()
    out = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert('RGB')
            r = model([tfm(img).cuda()])[0]
            keep = np.ones((img.height, img.width), bool)
            for l, b in zip(r['labels'], r['boxes']):
                x0, y0, x1, y1 = b.tolist()
                if names[l] in TRANSIENT and (x1 - x0) * (y1 - y0) > min_area * img.width * img.height:  # skip tiny false hits
                    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad  # generous margin: hair, shadows, motion blur
                    keep[max(0, int(y0 - dy)):int(y1 + dy) + 1, max(0, int(x0 - dx)):int(x1 + dx) + 1] = False
            out.append(keep)
    return out


def save(keep, path):
    Image.fromarray((keep * 255).astype(np.uint8)).save(path)


def mask_frames(frames_dir, colmap_mask_dir):
    """Detect transients in the selected frames; write COLMAP masks (<image name>.png). Returns #frames masked."""
    paths = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    keeps = detect(paths)
    hit = [k for k in keeps if not k.all()]
    if not hit:
        return 0
    Path(colmap_mask_dir).mkdir(parents=True, exist_ok=True)
    for p, k in zip(paths, keeps):
        save(k, Path(colmap_mask_dir) / f'{p.name}.png')
    return len(hit)


def attach_to_dataset(ns_dir, frames_dir, colmap_mask_dir):
    """After ns-process-data: copy masks next to the renamed images and reference them in transforms.json.
    ns-process-data copies the sorted source frames to images/frame_00001... in the same order."""
    import json
    src = sorted(p.name for p in Path(frames_dir).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    tf_path = Path(ns_dir) / 'transforms.json'
    tf = json.loads(tf_path.read_text())
    (Path(ns_dir) / 'masks').mkdir(exist_ok=True)
    for f in tf['frames']:
        i = int(Path(f['file_path']).stem.split('_')[-1]) - 1
        m = Path(colmap_mask_dir) / f'{src[i]}.png'
        dst = f'masks/{Path(f["file_path"]).stem}.png'
        if m.exists():
            Image.open(m).save(Path(ns_dir) / dst)
        else:  # nerfstudio needs a mask for every frame once any frame has one
            w, h = Image.open(Path(ns_dir) / f['file_path']).size
            save(np.ones((h, w), bool), Path(ns_dir) / dst)
        f['mask_path'] = dst
    tf_path.write_text(json.dumps(tf, indent=1))
