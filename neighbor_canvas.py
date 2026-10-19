"""Compose a new tile's immutable overlap from the visible world.

Coordinates, never names, define adjacency. Paint tiles using the exact same
dependency-first order as the browser export.
"""
import numpy as np
from PIL import Image, ImageDraw
from tile_grid import geometry, discover, ordered_regions, open_tile


def compose_lock(jobs, x, y, layout=None):
    layout = layout or geometry(jobs)
    width, height, overlap = (int(layout[k]) for k in ('width', 'height', 'overlap'))
    dx, dy = width - overlap, height - overlap
    px, py = int(x) * dx, int(y) * dy
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.full((height, width), 255, dtype=np.uint8)
    owners = np.zeros((height, width), dtype=np.uint16)
    intersections = []
    regs = discover(jobs)
    # Include the existing target in topological sorting (so the ordering
    # of subsequent dependents is stable), but skip its obsolete published art.
    for source in ordered_regions(regs):
        sx, sy = int(source['x']), int(source['y'])
        if (sx, sy) == (int(x), int(y)):
            continue
        wx, wy = sx * dx, sy * dy
        x0, x1 = max(px, wx), min(px + width, wx + width)
        y0, y1 = max(py, wy), min(py + height, wy + height)
        if x0 >= x1 or y0 >= y1:
            continue
        a0, a1, b0, b1 = x0 - px, x1 - px, y0 - py, y1 - py
        c0, c1, d0, d1 = x0 - wx, x1 - wx, y0 - wy, y1 - wy
        data = np.asarray(open_tile(source['image'], layout))
        canvas[b0:b1, a0:a1] = data[d0:d1, c0:c1]
        mask[b0:b1, a0:a1] = 0
        owners[b0:b1, a0:a1] = len(intersections) + 1
        intersections.append({'id': source['id'], 'grid': [sx, sy],
                              'target_rect': [a0, b0, a1, b1],
                              'world_rect': [x0, y0, x1, y1]})
    if not intersections:
        raise ValueError(f'No published tiles overlap grid ({x},{y})')
    return canvas, mask, owners, intersections


def save_debug_preview(path, canvas, mask, owners, intersections):
    base = canvas.copy()
    base[mask != 0] = (29, 38, 42)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    palette = ['#eac26e', '#72cbe5', '#e68b8b', '#98d6a4',
               '#cba5ec', '#e0a765', '#fff5bb', '#9dc9cb']
    for i, item in enumerate(intersections):
        x0, y0, x1, y1 = item['target_rect']
        color = palette[i % len(palette)]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=3)
        tx = x0 + 7 if x1 - x0 > 90 else max(0, x1 - 160)
        draw.text((tx, y0 + 7), item['id'], fill=color,
                  stroke_width=2, stroke_fill='#091215')
    image.save(path)
