"""Visible seam QA and region-neighbor diagnostic crops.

Scores detect long, unusual color/gradient jumps. They are pixel-based
proxies and do not prove hydrological or geographic continuity.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from seam_transition import _orient
from tile_grid import load_json


def analyze_seam(region, side, overlap, seam_transition=None, *, minimum_rows=16):
    """Measure persistent discontinuity at the actual adaptive cut.

    Compare pixel jumps across the cut against local natural image gradients.
    Strict mode rejects only pronounced, widespread discontinuity.
    """
    image = _orient(np.asarray(region, dtype=np.float32), side)
    h, w = image.shape[:2]
    core = w - int(overlap)
    if not (10 <= core < w - 4):
        return {'status': 'not_applicable', 'reason': 'Insufficient overlap'}
    ylo, yhi = max(0, int(overlap)), min(h, h - int(overlap))
    if yhi - ylo < minimum_rows:
        ylo, yhi = 0, h
    if yhi - ylo < minimum_rows:
        return {'status': 'not_applicable', 'reason': 'Insufficient seam length'}
    path = np.full(h, core, dtype=np.int32)
    if seam_transition and seam_transition.get('seam_path_px'):
        y0 = max(0, int(seam_transition.get('seam_y_start', 0)))
        entries = np.asarray(seam_transition['seam_path_px'], dtype=np.int32)
        count = min(len(entries), h - y0)
        if count:
            path[y0:y0 + count] = entries[:count]
    rows = np.arange(ylo, yhi, dtype=np.int32)
    boundary = np.clip(path[rows], 10, w - 10)
    jumps = np.abs(image[rows, boundary] - image[rows, boundary - 1]).mean(axis=1)
    a = np.abs(image[rows, boundary - 8] - image[rows, boundary - 9]).mean(axis=1)
    b = np.abs(image[rows, boundary + 8] - image[rows, boundary + 7]).mean(axis=1)
    texture = (a + b) / 2
    excess = np.maximum(0, jumps - texture)
    persistent = float(np.mean((jumps > 25) & (excess > 14)))
    severe = ((float(jumps.mean()) > 28 and float(excess.mean()) > 17 and persistent > .60)
              or (float(jumps.mean()) > 50 and persistent > .75))
    return {
        'status': 'rejected' if severe else 'passed',
        'side': side, 'sample_rows': int(len(rows)),
        'edge_color_delta': round(float(jumps.mean()), 3),
        'edge_gradient_delta': round(float(excess.mean()), 3),
        'texture_gradient': round(float(texture.mean()), 3),
        'discontinuity_fraction': round(persistent, 4),
        'feature_continuity_score': round(float(np.clip(1 - excess.mean() / 35, 0, 1)), 4),
        'metric_note': 'Gradient-based proxy; not semantic terrain recognition',
    }


def save_seam_previews(world_preview, metadata_file, output_dir, *, radius=240):
    """Crop the composited draft around its cardinal and diagonal joins."""
    report = load_json(metadata_file)
    x0, y0, x1, y1 = report['target_rect']
    ov = int(report['overlap_px'])
    side = report['primary_side']
    if side == 'east':
        x = x1 - ov
        joins = [('east', x, (y0+y1)//2), ('ne', x, y0), ('se', x, y1-ov)]
    elif side == 'west':
        x = x0 + ov
        joins = [('west', x, (y0+y1)//2), ('nw', x, y0), ('sw', x, y1-ov)]
    elif side == 'north':
        y = y0 + ov
        joins = [('north', (x0+x1)//2, y), ('nw', x0, y), ('ne', x1-ov, y)]
    else:
        y = y1 - ov
        joins = [('south', (x0+x1)//2, y), ('sw', x0, y), ('se', x1-ov, y)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    with Image.open(world_preview) as world:
        for name, cx, cy in joins:
            r = max(32, int(radius))
            xa, ya = max(0, cx-r), max(0, cy-r)
            xb, yb = min(world.width, cx+r), min(world.height, cy+r)
            crop = world.crop((xa, ya, xb, yb)).convert('RGB')
            draw = ImageDraw.Draw(crop)
            guide = (244, 199, 96)
            if side in ('east', 'west'):
                lx = cx-xa
                draw.line([(lx, 0), (lx, crop.height-1)], fill=guide, width=2)
            else:
                ly = cy-ya
                draw.line([(0,ly), (crop.width-1,ly)], fill=guide, width=2)
            draw.rectangle((0, 0, 175, 22), fill=(16, 29, 36))
            draw.text((6, 5), f'{name.upper()} / WORLD', fill=guide)
            path = output_dir / f'seam_{name}_preview.png'
            crop.save(path)
            saved[name] = str(path)
    return saved
