"""Preview a draft exactly where it would appear in the exported world.

Only source tiles whose image rectangle intersects a 3x3-cell viewport are
loaded. The preview uses the same dependency-first ordering as the web atlas.
It never writes to published tiles or their metadata.
"""
from pathlib import Path

from PIL import Image

from tile_grid import discover, geometry, open_tile, ordered_regions, save_json


def draft_dependencies(meta):
    """Every pre-existing source that contributed fixed terrain must precede this tile."""
    deps = []
    for side_data in meta.get('neighbors', {}).values():
        name = side_data['id']
        if name not in deps:
            deps.append(name)
    for diagonal in meta.get('diagonal_neighbors', {}).values():
        name = diagonal['id']
        if name not in deps:
            deps.append(name)
    for intersection in meta.get('overlap_intersections', []):
        name = intersection['id']
        if name not in deps:
            deps.append(name)
    return deps


def create_world_preview(job, draft_image, meta, layout=None):
    """Make a 3x3-cell local crop of the prospective published world.

    It substitutes the draft for an already-published target, or inserts the
    target if new. If the prospective dependency graph is invalid, fail rather
    than showing a misleading paste order.
    """
    job = Path(job).resolve()
    draft_image = Path(draft_image).resolve()
    jobs = job.parent
    layout = layout or geometry(jobs)
    w, h, ov = (int(layout[k]) for k in ('width', 'height', 'overlap'))
    step_x, step_y = w - ov, h - ov
    tx, ty = int(meta['target']['x']), int(meta['target']['y'])
    target_id = meta['region_id']

    regions = discover(jobs)
    old = regions.get((tx, ty))
    if old is not None and old['id'] != target_id:
        raise ValueError('Draft target conflicts with a different published tile')
    deps = draft_dependencies(meta)
    if target_id in deps:
        raise ValueError('Draft cannot depend on itself')
    regions[(tx, ty)] = {
        'id': target_id, 'x': tx, 'y': ty,
        'image': str(draft_image), 'dependencies': deps,
    }
    try:
        order = ordered_regions(regions)
    except ValueError as exc:
        raise ValueError(
            f'Cannot preview draft in actual export paint order: {exc}'
        ) from exc

    # Center on the target and include one logical cell in every direction.
    x0, y0 = (tx - 1) * step_x, (ty - 1) * step_y
    x1, y1 = (tx + 1) * step_x + w, (ty + 1) * step_y + h
    image = Image.new('RGB', (x1 - x0, y1 - y0), (12, 24, 29))
    painted = []
    for tile in order:
        tile_x = int(tile['x']) * step_x
        tile_y = int(tile['y']) * step_y
        left, top = max(x0, tile_x), max(y0, tile_y)
        right, bottom = min(x1, tile_x + w), min(y1, tile_y + h)
        if right <= left or bottom <= top:
            continue
        with open_tile(tile['image'], layout) as source:
            cut = source.crop((left - tile_x, top - tile_y,
                               right - tile_x, bottom - tile_y))
        image.paste(cut, (left - x0, top - y0))
        painted.append(tile['id'])

    out = job / 'draft'
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'world_preview.png'
    image.save(destination)
    save_json(out / 'world_preview.json', {
        'format': 'frontier_draft_world_preview_v1',
        'target': {'id': target_id, 'x': tx, 'y': ty},
        'world_rect': [x0, y0, x1, y1],
        'target_rect': [tx * step_x - x0, ty * step_y - y0,
                        tx * step_x - x0 + w, ty * step_y - y0 + h],
        'render_order': 'dependencies_first',
        'dependencies': deps,
        'painted': painted,
        'pixels': [image.width, image.height],
    })
    return destination
