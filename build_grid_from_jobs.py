#!/usr/bin/env python3
"""Compose all local tiles into one world, exclusively under _grid."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from tile_grid import geometry, discover, ordered_regions, open_tile, save_json


def _cost(a, b, edge_penalty_strength=55.0, edge_margin_ratio=0.18):
    """Pixel difference cost with a soft penalty near overlap edges.

    Minimum-cost seams that are allowed to touch x=0/x=overlap-1 tend to recreate
    a hard tile boundary. The penalty keeps the seam inside the overlap while
    still allowing it to move away from the center when image content strongly
    prefers another path.
    """
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).sum(axis=2)
    h, w = diff.shape

    if w <= 2:
        return diff

    x = np.arange(w, dtype=np.float32)
    center = (w - 1) * 0.5
    half = max(center, 1.0)

    # 0 in the middle, 1 at the outer overlap boundaries.
    edge_distance = np.abs(x - center) / half

    # Mostly leave the central region untouched and increase the cost sharply
    # toward both edges. The fourth power prevents over-constraining the seam.
    penalty = edge_distance ** 4

    # Add an extra soft wall in the outer margin so the seam cannot casually
    # ride exactly along the tile boundary.
    margin = max(2, int(round(w * edge_margin_ratio)))
    if margin > 0:
        t = np.linspace(1.0, 0.0, margin, dtype=np.float32)
        wall = (t * t * (3.0 - 2.0 * t))
        penalty[:margin] += wall
        penalty[-margin:] += wall[::-1]

    # Scale penalty relative to a typical RGB absolute-difference magnitude.
    return diff + penalty[None, :] * float(edge_penalty_strength)


def _min_vertical_seam(cost):
    """Minimum-cost top-to-bottom seam through a 2D cost image."""
    h, w = cost.shape
    dp = cost.astype(np.float32).copy()
    parent = np.zeros((h, w), np.int32)
    inf = np.float32(1e20)
    xs = np.arange(w)

    for y in range(1, h):
        prev = dp[y - 1]
        left = np.empty_like(prev)
        right = np.empty_like(prev)
        left[0] = inf
        left[1:] = prev[:-1]
        right[-1] = inf
        right[:-1] = prev[1:]

        stack = np.stack([left, prev, right])
        choice = np.argmin(stack, axis=0)
        parent[y] = np.clip(xs + choice - 1, 0, w - 1)
        dp[y] = cost[y] + stack[choice, xs]

    seam = np.zeros(h, np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for y in range(h - 1, 0, -1):
        seam[y - 1] = parent[y, seam[y]]
    return seam


def _merge_lr(left, right, feather, edge_penalty_strength=55.0):
    """Narrow seam: left owns one side, right owns the other."""
    seam = _min_vertical_seam(
        _cost(left, right, edge_penalty_strength=edge_penalty_strength)
    )
    h, w, _ = left.shape
    out = np.empty_like(left)

    for y, x in enumerate(seam):
        x = int(x)
        a = max(0, x - feather)
        b = min(w, x + feather + 1)

        out[y, :a] = left[y, :a]
        out[y, b:] = right[y, b:]

        if b > a:
            alpha = np.linspace(0.0, 1.0, b - a, dtype=np.float32)[:, None]
            mixed = left[y, a:b].astype(np.float32) * (1.0 - alpha)
            mixed += right[y, a:b].astype(np.float32) * alpha
            out[y, a:b] = np.clip(mixed, 0, 255).astype(np.uint8)

    return out, seam


def _merge_tb(top, bottom, feather, edge_penalty_strength=55.0):
    merged, seam = _merge_lr(
        np.transpose(top, (1, 0, 2)),
        np.transpose(bottom, (1, 0, 2)),
        feather,
        edge_penalty_strength=edge_penalty_strength,
    )
    return np.transpose(merged, (1, 0, 2)), seam


def _soft_factor(values, seam, feather, greater_is_tile):
    """Per-pixel tile ownership around a 1D seam.

    Returns 0 for the already-built world side, 1 for the incoming tile side,
    with a narrow linear transition of roughly 2*feather+1 pixels.
    """
    feather = max(0, int(feather))
    if feather == 0:
        if greater_is_tile:
            return (values > seam).astype(np.float32)
        return (values < seam).astype(np.float32)

    width = float(2 * feather + 1)
    if greater_is_tile:
        factor = (values - seam + feather + 0.5) / width
    else:
        factor = (seam - values + feather + 0.5) / width
    return np.clip(factor, 0.0, 1.0).astype(np.float32)


def _reconcile_corner(tile, world_patch, corner, overlap, feather, seams):
    """Combine two narrow side seams into one 2D corner ownership mask."""
    if overlap <= 0:
        return

    if corner == 'northwest':
        west = seams.get('west')
        north = seams.get('north')
        if west is None or north is None:
            return
        current = tile[:overlap, :overlap].astype(np.float32)
        existing = world_patch[:overlap, :overlap].astype(np.float32)
        yy, xx = np.indices((overlap, overlap), dtype=np.float32)
        fx = _soft_factor(xx, west[:overlap, None], feather, True)
        fy = _soft_factor(yy, north[None, :overlap], feather, True)
        alpha = fx * fy
        tile[:overlap, :overlap] = np.clip(
            existing * (1.0 - alpha[:, :, None]) + current * alpha[:, :, None],
            0, 255
        ).astype(np.uint8)

    elif corner == 'northeast':
        east = seams.get('east')
        north = seams.get('north')
        if east is None or north is None:
            return
        current = tile[:overlap, -overlap:].astype(np.float32)
        existing = world_patch[:overlap, -overlap:].astype(np.float32)
        yy, xx = np.indices((overlap, overlap), dtype=np.float32)
        fx = _soft_factor(xx, east[:overlap, None], feather, False)
        fy = _soft_factor(yy, north[None, -overlap:], feather, True)
        alpha = fx * fy
        tile[:overlap, -overlap:] = np.clip(
            existing * (1.0 - alpha[:, :, None]) + current * alpha[:, :, None],
            0, 255
        ).astype(np.uint8)

    elif corner == 'southwest':
        west = seams.get('west')
        south = seams.get('south')
        if west is None or south is None:
            return
        current = tile[-overlap:, :overlap].astype(np.float32)
        existing = world_patch[-overlap:, :overlap].astype(np.float32)
        yy, xx = np.indices((overlap, overlap), dtype=np.float32)
        fx = _soft_factor(xx, west[-overlap:, None], feather, True)
        fy = _soft_factor(yy, south[None, :overlap], feather, False)
        alpha = fx * fy
        tile[-overlap:, :overlap] = np.clip(
            existing * (1.0 - alpha[:, :, None]) + current * alpha[:, :, None],
            0, 255
        ).astype(np.uint8)

    elif corner == 'southeast':
        east = seams.get('east')
        south = seams.get('south')
        if east is None or south is None:
            return
        current = tile[-overlap:, -overlap:].astype(np.float32)
        existing = world_patch[-overlap:, -overlap:].astype(np.float32)
        yy, xx = np.indices((overlap, overlap), dtype=np.float32)
        fx = _soft_factor(xx, east[-overlap:, None], feather, False)
        fy = _soft_factor(yy, south[None, -overlap:], feather, False)
        alpha = fx * fy
        tile[-overlap:, -overlap:] = np.clip(
            existing * (1.0 - alpha[:, :, None]) + current * alpha[:, :, None],
            0, 255
        ).astype(np.uint8)


def _compose_tile(world_rgb, occupied, tile, px, py, pos, placed, layout, feather, edge_penalty_strength):
    """Compose one tile with narrow minimum-cost seams against placed neighbors."""
    w, h, ov = layout['width'], layout['height'], layout['overlap']
    x, y = pos

    tile_original = tile.copy()
    composed = tile.copy()
    seams = {}

    ys = slice(py, py + h)
    xs = slice(px, px + w)
    world_patch = world_rgb[ys, xs].copy()

    # Direct neighbors already present in the world.
    if (x - 1, y) in placed:  # west
        existing = world_patch[:, :ov]
        merged, seam = _merge_lr(existing, composed[:, :ov], feather, edge_penalty_strength)
        composed[:, :ov] = merged
        seams['west'] = seam

    if (x + 1, y) in placed:  # east
        existing = world_patch[:, -ov:]
        merged, seam = _merge_lr(composed[:, -ov:], existing, feather, edge_penalty_strength)
        composed[:, -ov:] = merged
        seams['east'] = seam

    if (x, y - 1) in placed:  # north
        existing = world_patch[:ov]
        merged, seam = _merge_tb(existing, composed[:ov], feather, edge_penalty_strength)
        composed[:ov] = merged
        seams['north'] = seam

    if (x, y + 1) in placed:  # south
        existing = world_patch[-ov:]
        merged, seam = _merge_tb(composed[-ov:], existing, feather, edge_penalty_strength)
        composed[-ov:] = merged
        seams['south'] = seam

    # Rebuild the doubly-overlapped corner from both seam ownership functions.
    # Use the unmodified incoming tile there, so sequential side merges cannot
    # smear the same corner twice.
    corner_tile = tile_original.copy()
    corner_world = world_patch

    if 'west' in seams and 'north' in seams:
        _reconcile_corner(corner_tile, corner_world, 'northwest', ov, feather, seams)
        composed[:ov, :ov] = corner_tile[:ov, :ov]
    if 'east' in seams and 'north' in seams:
        _reconcile_corner(corner_tile, corner_world, 'northeast', ov, feather, seams)
        composed[:ov, -ov:] = corner_tile[:ov, -ov:]
    if 'west' in seams and 'south' in seams:
        _reconcile_corner(corner_tile, corner_world, 'southwest', ov, feather, seams)
        composed[-ov:, :ov] = corner_tile[-ov:, :ov]
    if 'east' in seams and 'south' in seams:
        _reconcile_corner(corner_tile, corner_world, 'southeast', ov, feather, seams)
        composed[-ov:, -ov:] = corner_tile[-ov:, -ov:]

    # If only a diagonal tile exists in some corner, do not blindly overwrite
    # that already-owned square. Side-connected cells will resolve it later.
    existing_mask = occupied[ys, xs]
    handled = np.zeros((h, w), dtype=bool)
    if 'west' in seams:
        handled[:, :ov] = True
    if 'east' in seams:
        handled[:, -ov:] = True
    if 'north' in seams:
        handled[:ov, :] = True
    if 'south' in seams:
        handled[-ov:, :] = True

    write = (~existing_mask) | handled
    patch = world_rgb[ys, xs]
    patch[write] = composed[write]
    world_rgb[ys, xs] = patch
    occupied[ys, xs] |= True

    return seams




def build(jobs, feather=6, edge_penalty_strength=55.0):
    jobs = jobs.resolve()
    layout = geometry(jobs)
    regions = discover(jobs)
    if not regions:
        raise ValueError('No tiles found')
    if feather < 0:
        raise ValueError('feather must be non-negative')

    ordered = ordered_regions(regions)
    w, h, ov = layout['width'], layout['height'], layout['overlap']
    dx, dy = w - ov, h - ov

    min_x = min(x for x, y in regions)
    max_x = max(x for x, y in regions)
    min_y = min(y for x, y in regions)
    max_y = max(y for x, y in regions)
    size = ((max_x - min_x) * dx + w, (max_y - min_y) * dy + h)

    # Validate everything first so a failed build cannot overwrite the previous export.
    for r in ordered:
        if r.get('layout', layout) != layout:
            raise ValueError(f"{r['id']}: conflicting grid geometry")
        with open_tile(r['image'], layout):
            pass

    world_w, world_h = size
    world_rgb = np.zeros((world_h, world_w, 3), dtype=np.uint8)
    occupied = np.zeros((world_h, world_w), dtype=bool)
    placed = set()
    placements = []

    for r in ordered:
        px = (r['x'] - min_x) * dx
        py = (r['y'] - min_y) * dy
        pos = (r['x'], r['y'])

        with open_tile(r['image'], layout) as im:
            tile = np.asarray(im, dtype=np.uint8).copy()

        seams = _compose_tile(
            world_rgb, occupied, tile, px, py, pos, placed, layout, feather,
            edge_penalty_strength
        )
        placed.add(pos)

        placements.append({
            'id': r['id'],
            'x': r['x'],
            'y': r['y'],
            'pixel_x': px,
            'pixel_y': py,
            'image': str(Path(r['image']).relative_to(jobs)),
            'seams_with_placed_neighbors': sorted(seams.keys()),
        })

    alpha = np.zeros((world_h, world_w), dtype=np.uint8)
    alpha[occupied] = 255
    world = Image.fromarray(np.dstack([world_rgb, alpha]), 'RGBA')

    out = jobs / '_grid'
    out.mkdir(exist_ok=True)

    temp = out / 'world_final.tmp.png'
    world.save(temp)
    temp.replace(out / 'world_final.png')

    preview = ImageOps.contain(world, (1800, 1200), Image.Resampling.LANCZOS)
    bg = Image.new('RGBA', preview.size, (18, 17, 15, 255))
    bg.alpha_composite(preview)
    bg.convert('RGB').save(out / 'grid_preview.png')

    save_json(out / 'grid.json', {
        'format': 'local_tile_v1',
        'compositor': 'narrow_min_cost_seam_v2_edge_penalty',
        'feather': feather,
        'edge_penalty_strength': edge_penalty_strength,
        'layout': layout,
        'size': list(size),
        'origin': [-min_x * dx, -min_y * dy],
        'regions': placements,
    })

    print(
        f'Built {len(ordered)} tiles with narrow seams '
        f'(feather={feather}px, edge_penalty={edge_penalty_strength:g}): '
        f'{size[0]}x{size[1]}'
    )
    print(out / 'world_final.png')
    return out / 'world_final.png'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', default='jobs')
    p.add_argument('--feather', type=int, default=6,
                   help='Half-width of narrow seam blending in pixels (default: 6)')
    p.add_argument('--edge-penalty', type=float, default=55.0,
                   help='Penalty that keeps seams away from overlap edges (default: 55)')
    p.add_argument('--force', action='store_true',
                   help='Compatibility option; export is always rebuilt.')
    args = p.parse_args()
    build(Path(args.jobs), feather=args.feather,
          edge_penalty_strength=args.edge_penalty)


if __name__ == '__main__':
    main()
