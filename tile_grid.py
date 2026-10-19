"""Shared geometry and tile discovery; no cumulative images in region jobs."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

DIRS = {'north': (0, -1), 'east': (1, 0), 'south': (0, 1), 'west': (-1, 0)}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def geometry(jobs, width=None, height=None, overlap=None):
    path = jobs / 'layout.json'
    if path.exists():
        result = load_json(path)
    else:
        with Image.open(jobs / 'center' / 'region.png') as im:
            w, h = im.size
        result = {'version': 1, 'width': w, 'height': h,
                  'overlap': min(256, min(w, h) // 3) if overlap is None else overlap}
    for name, value in [('width', width), ('height', height), ('overlap', overlap)]:
        if value is not None and result[name] != value:
            raise ValueError(f'{name}={value} conflicts with grid geometry {result[name]}')
    w, h, ov = result['width'], result['height'], result['overlap']
    if not (w > 0 and h > 0 and 0 < ov < min(w, h)):
        raise ValueError(f'Invalid grid geometry: {result}')
    return result


def discover(jobs):
    found = {}
    center = jobs / 'center' / 'region.png'
    if center.exists():
        found[(0, 0)] = {'id': 'center', 'x': 0, 'y': 0,
                         'image': str(center.resolve()), 'job': str(center.parent.resolve()),
                         'dependencies': []}
    for d in sorted(jobs.iterdir()):
        if not d.is_dir() or d.name.startswith('_') or d.name == 'center':
            continue
        path = d / 'region.json'
        if not path.exists():
            continue
        meta = load_json(path)
        if meta.get('format') != 'local_tile_v1':
            raise ValueError(f'{path}: legacy region; migrate or archive before using the tile grid')
        image = Path(meta['image'])
        if not image.is_absolute():
            image = d / image
        if not image.is_file():
            raise FileNotFoundError(image)
        pos = (int(meta['x']), int(meta['y']))
        if pos in found:
            raise ValueError(f'Duplicate grid position {pos}: {d}')
        found[pos] = {**meta, 'image': str(image.resolve()), 'job': str(d.resolve())}
    ids = [r['id'] for r in found.values()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate region IDs')
    return found


def neighbors(regions, x, y):
    return {s: regions[(x + dx, y + dy)] for s, (dx, dy) in DIRS.items()
            if (x + dx, y + dy) in regions}


def open_tile(path, layout):
    with Image.open(path) as im:
        if im.size != (layout['width'], layout['height']):
            raise ValueError(f'{path}: tile size {im.size} differs from grid geometry')
        return im.convert('RGB')


def ordered_regions(regions):
    remaining = {r['id']: r for r in regions.values()}
    ordered, done = [], set()
    while remaining:
        ready = sorted(rid for rid, r in remaining.items()
                       if set(r.get('dependencies', [])) <= done)
        if not ready:
            raise ValueError('Missing tile dependencies or dependency cycle')
        for rid in ready:
            ordered.append(remaining.pop(rid))
            done.add(rid)
    return ordered
