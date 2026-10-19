#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from tile_grid import load_json, save_json, open_tile, discover
from terrain_alignment import align_terrain, AlignmentError


def resolve_job_and_generated(job, generated=None, jobs='jobs'):
    jobs = Path(jobs).resolve()
    job = Path(job)
    if not job.is_absolute() and len(job.parts) == 1:
        job = jobs / job
    else:
        job = job.resolve()
    if generated is None:
        generated = job / 'generated.png'
    else:
        generated = Path(generated)
        if not generated.is_absolute():
            generated = generated.resolve()
    return job, generated



def cost(a, b):
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).sum(axis=2)


def min_v(c):
    h, w = c.shape
    dp = c.copy()
    par = np.zeros((h, w), np.int32)
    big = np.float32(1e20)
    xs = np.arange(w)
    for y in range(1, h):
        prev = dp[y - 1]
        l = np.empty_like(prev)
        r = np.empty_like(prev)
        l[0] = big
        l[1:] = prev[:-1]
        r[-1] = big
        r[:-1] = prev[1:]
        st = np.stack([l, prev, r])
        idx = np.argmin(st, axis=0)
        par[y] = np.clip(xs + idx - 1, 0, w - 1)
        dp[y] = c[y] + st[idx, xs]
    s = np.zeros(h, np.int32)
    s[-1] = int(np.argmin(dp[-1]))
    for y in range(h - 1, 0, -1):
        s[y - 1] = par[y, s[y]]
    return s


def merge_lr(l, r, feather):
    s = min_v(cost(l, r))
    h, w, _ = l.shape
    out = np.empty_like(l)
    for y, x in enumerate(s):
        x = int(x)
        a = max(0, x - feather)
        b = min(w, x + feather + 1)
        out[y, :a] = l[y, :a]
        out[y, b:] = r[y, b:]
        if b > a:
            al = np.linspace(0, 1, b - a, dtype=np.float32)[:, None]
            out[y, a:b] = np.clip(l[y, a:b] * (1 - al) + r[y, a:b] * al, 0, 255).astype(np.uint8)
    return out, s


def merge_tb(t, b, feather):
    # Reuse vertical seam logic by transposing spatial axes.
    mt, s = merge_lr(np.transpose(t, (1, 0, 2)), np.transpose(b, (1, 0, 2)), feather)
    return np.transpose(mt, (1, 0, 2)), s


def _guarded_cost(a, b, edge_penalty=36.0):
    """RGB mismatch cost with a mild preference for seams inside the overlap."""
    c = cost(a, b)
    w = c.shape[1]
    if w <= 2:
        return c

    x = np.arange(w, dtype=np.float32)
    center = (w - 1) * 0.5
    half = max(center, 1.0)
    d = np.abs(x - center) / half

    # Very weak in the middle, increasingly expensive near the two boundaries.
    # This avoids a seam that collapses onto the physical tile edge unless the
    # image evidence strongly prefers it.
    penalty = (d ** 4) * float(edge_penalty)
    return c + penalty[None, :]


def merge_lr_hard(l, r, edge_penalty=36.0):
    """Content-aware hard cut: every pixel comes from exactly one source."""
    s = min_v(_guarded_cost(l, r, edge_penalty=edge_penalty))
    h, w, _ = l.shape
    out = np.empty_like(l)

    for y, x in enumerate(s):
        x = int(np.clip(x, 0, w - 1))
        # Left source owns [0, x), right source owns [x, w).
        # No RGB interpolation -> no ghosted houses/trees.
        out[y, :x] = l[y, :x]
        out[y, x:] = r[y, x:]

    return out, s


def merge_tb_hard(t, b, edge_penalty=36.0):
    """Top/bottom version of merge_lr_hard()."""
    mt, s = merge_lr_hard(
        np.transpose(t, (1, 0, 2)),
        np.transpose(b, (1, 0, 2)),
        edge_penalty=edge_penalty,
    )
    return np.transpose(mt, (1, 0, 2)), s



def _side_corner_slices(side, h, w, overlap):
    ov = int(overlap)
    if side == 'west':
        return {
            'northwest': (slice(0, ov), slice(0, ov)),
            'southwest': (slice(h - ov, h), slice(0, ov)),
        }
    if side == 'east':
        return {
            'northeast': (slice(0, ov), slice(w - ov, w)),
            'southeast': (slice(h - ov, h), slice(w - ov, w)),
        }
    if side == 'north':
        return {
            'northwest': (slice(0, ov), slice(0, ov)),
            'northeast': (slice(0, ov), slice(w - ov, w)),
        }
    if side == 'south':
        return {
            'southwest': (slice(h - ov, h), slice(0, ov)),
            'southeast': (slice(h - ov, h), slice(w - ov, w)),
        }
    raise ValueError(f'Unknown side: {side}')


def _merge_side(region, neighbor, side, overlap, feather, hard=False):
    ov = int(overlap)
    candidate = region.copy()

    if side == 'west':
        if hard:
            candidate[:, :ov], seam = merge_lr_hard(
                neighbor[:, -ov:], region[:, :ov]
            )
        else:
            candidate[:, :ov], seam = merge_lr(
                neighbor[:, -ov:], region[:, :ov], feather
            )

    elif side == 'east':
        if hard:
            candidate[:, -ov:], seam = merge_lr_hard(
                region[:, -ov:], neighbor[:, :ov]
            )
        else:
            candidate[:, -ov:], seam = merge_lr(
                region[:, -ov:], neighbor[:, :ov], feather
            )

    elif side == 'north':
        if hard:
            candidate[:ov], seam = merge_tb_hard(
                neighbor[-ov:], region[:ov]
            )
        else:
            candidate[:ov], seam = merge_tb(
                neighbor[-ov:], region[:ov], feather
            )

    elif side == 'south':
        if hard:
            candidate[-ov:], seam = merge_tb_hard(
                region[-ov:], neighbor[:ov]
            )
        else:
            candidate[-ov:], seam = merge_tb(
                region[-ov:], neighbor[:ov], feather
            )

    else:
        raise ValueError(f'Unknown side: {side}')

    return candidate, seam


def merge_sides_without_corner_overwrite(region, sides, neighbor_files, job, layout, feather):
    """Merge side strips deterministically without later sides erasing corners.

    The primary side is first in *sides* and owns its complete strip, including
    both corners. Later cardinal merges may update their strips, but corners
    already touched by an earlier side are restored from the pre-merge image.
    Diagonal reconciliation runs afterwards and remains the final authority for
    physical overlap×overlap diagonal corners.
    """
    h = int(layout['height'])
    w = int(layout['width'])
    ov = int(layout['overlap'])

    protected_corners = set()
    reports = []

    for index, side in enumerate(sides):
        neighbor = np.array(open_tile(job / neighbor_files[side], layout))
        before = region.copy()

        is_primary = index == 0
        candidate, seam = _merge_side(
            before, neighbor, side, ov, feather,
            hard=is_primary,
        )

        protected_here = []
        corners = _side_corner_slices(side, h, w, ov)
        for corner, (ys, xs) in corners.items():
            if corner in protected_corners:
                candidate[ys, xs] = before[ys, xs]
                protected_here.append(corner)

        region[:] = candidate
        protected_corners.update(corners.keys())

        reports.append({
            'side': side,
            'order': index,
            'primary': is_primary,
            'method': 'content_aware_hard_seam' if is_primary else 'soft_min_cost_seam',
            'seam_min_px': int(np.min(seam)),
            'seam_max_px': int(np.max(seam)),
            'seam_mean_px': float(np.mean(seam)),
            'protected_from_overwrite': sorted(protected_here),
        })

    return reports



def soften_primary_overlap(region, primary, side, overlap, transition=64, exact_band=4):
    """Blend the published primary overlap toward the existing neighbor.

    The inner edge of the overlap stays close to the generated/aligned tile,
    while the outer edge becomes pixel-identical to the existing neighbor.
    This avoids a hard rectangular transition at the overlap/core boundary.
    """
    transition = max(1, min(int(transition), overlap))
    exact_band = max(1, min(int(exact_band), transition))

    if side == 'west':
        current = region[:, :overlap].astype(np.float32)
        target = primary[:, -overlap:].astype(np.float32)
        # Outer/shared edge is x=0 for a west neighbor.
        d = np.arange(overlap, dtype=np.float32)
        alpha = np.clip((transition - d) / transition, 0.0, 1.0)
        alpha[d < exact_band] = 1.0
        alpha = alpha[None, :, None]
        region[:, :overlap] = np.clip(current * (1 - alpha) + target * alpha, 0, 255).astype(np.uint8)

    elif side == 'east':
        current = region[:, -overlap:].astype(np.float32)
        target = primary[:, :overlap].astype(np.float32)
        # Outer/shared edge is x=overlap-1 for an east neighbor.
        d = np.arange(overlap - 1, -1, -1, dtype=np.float32)
        alpha = np.clip((transition - d) / transition, 0.0, 1.0)
        alpha[d < exact_band] = 1.0
        alpha = alpha[None, :, None]
        region[:, -overlap:] = np.clip(current * (1 - alpha) + target * alpha, 0, 255).astype(np.uint8)

    elif side == 'north':
        current = region[:overlap].astype(np.float32)
        target = primary[-overlap:].astype(np.float32)
        # Outer/shared edge is y=0 for a north neighbor.
        d = np.arange(overlap, dtype=np.float32)
        alpha = np.clip((transition - d) / transition, 0.0, 1.0)
        alpha[d < exact_band] = 1.0
        alpha = alpha[:, None, None]
        region[:overlap] = np.clip(current * (1 - alpha) + target * alpha, 0, 255).astype(np.uint8)

    else:  # south
        current = region[-overlap:].astype(np.float32)
        target = primary[:overlap].astype(np.float32)
        # Outer/shared edge is y=overlap-1 for a south neighbor.
        d = np.arange(overlap - 1, -1, -1, dtype=np.float32)
        alpha = np.clip((transition - d) / transition, 0.0, 1.0)
        alpha[d < exact_band] = 1.0
        alpha = alpha[:, None, None]
        region[-overlap:] = np.clip(current * (1 - alpha) + target * alpha, 0, 255).astype(np.uint8)

    return transition, exact_band



def _corner_patch(image, corner, overlap):
    """Return one overlap×overlap corner patch from an RGB ndarray."""
    if corner == 'northwest':
        return image[:overlap, :overlap]
    if corner == 'northeast':
        return image[:overlap, -overlap:]
    if corner == 'southwest':
        return image[-overlap:, :overlap]
    if corner == 'southeast':
        return image[-overlap:, -overlap:]
    raise ValueError(f'Unknown corner: {corner}')


def _to_nw_corner(patch, corner):
    """Orient any corner patch so its physical outer corner is top-left."""
    if corner == 'northwest':
        return patch.copy()
    if corner == 'northeast':
        return np.fliplr(patch).copy()
    if corner == 'southwest':
        return np.flipud(patch).copy()
    if corner == 'southeast':
        return np.flipud(np.fliplr(patch)).copy()
    raise ValueError(f'Unknown corner: {corner}')


def _from_nw_corner(patch, corner):
    """Undo _to_nw_corner()."""
    if corner == 'northwest':
        return patch
    if corner == 'northeast':
        return np.fliplr(patch)
    if corner == 'southwest':
        return np.flipud(patch)
    if corner == 'southeast':
        return np.fliplr(np.flipud(patch))
    raise ValueError(f'Unknown corner: {corner}')


def _corner_difference_cost(current, target):
    """Structure-aware RGB cost for choosing an unobtrusive hard seam."""
    a = current.astype(np.float32)
    b = target.astype(np.float32)

    # RGB mismatch.
    rgb = np.abs(a - b).mean(axis=2)

    # Add a small gradient mismatch term so the seam prefers crossing regions
    # with similar edges instead of cutting directly through houses/trees/roads.
    ga = a.mean(axis=2)
    gb = b.mean(axis=2)

    def grad(gray):
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
        return np.hypot(gx, gy)

    edge = np.abs(grad(ga) - grad(gb))
    return rgb + 0.35 * edge


def _minimum_corner_path(cost, extent):
    """Find a monotone low-cost path from top edge to left edge.

    In canonical NW orientation the diagonal tile owns the physical outer
    corner (top-left), while the current tile owns the inner/right/bottom part.
    The path runs from (0, extent) to (extent, 0), moving only down or left.
    """
    extent = int(max(2, min(extent, min(cost.shape) - 1)))
    sub = cost[:extent + 1, :extent + 1].astype(np.float32)

    inf = np.float32(1e20)
    dp = np.full_like(sub, inf)
    parent = np.full((extent + 1, extent + 1), -1, dtype=np.int8)

    # Start on the top edge, away from the physical corner.
    dp[0, extent] = sub[0, extent]

    # We may only move down or left. Iterate by Manhattan distance from start.
    for d in range(1, 2 * extent + 1):
        y_min = max(0, d - extent)
        y_max = min(extent, d)
        for y in range(y_min, y_max + 1):
            x = extent - (d - y)
            if x < 0 or x > extent:
                continue

            best = inf
            p = -1

            # Arrived from above: previous point (y-1, x).
            if y > 0 and dp[y - 1, x] < best:
                best = dp[y - 1, x]
                p = 0

            # Arrived from the right: previous point (y, x+1).
            if x < extent and dp[y, x + 1] < best:
                best = dp[y, x + 1]
                p = 1

            if p >= 0:
                dp[y, x] = best + sub[y, x]
                parent[y, x] = p

    # Backtrack from the left edge.
    y, x = extent, 0
    path = [(y, x)]
    while not (y == 0 and x == extent):
        p = int(parent[y, x])
        if p == 0:
            y -= 1
        elif p == 1:
            x += 1
        else:
            raise RuntimeError('Could not reconstruct diagonal corner seam')
        path.append((y, x))
    path.reverse()
    return path


def _cloud_presence_penalty(a, b):
    """Penalty for placing a hard seam through visible cloud/fog.

    We do NOT blend clouds. Instead, the minimum-cost seam is encouraged to
    route around bright, neutral, low-saturation cloud masses when a reasonable
    terrain route exists. This preserves the previously good hard mountain seam
    and avoids reintroducing ghosting.
    """
    af = a.astype(np.float32)
    bf = b.astype(np.float32)

    def luminance(x):
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    def saturation_range(x):
        return x.max(axis=2) - x.min(axis=2)

    ya = luminance(af)
    yb = luminance(bf)
    sa = saturation_range(af)
    sb = saturation_range(bf)

    # Bright + nearly neutral in either source is likely cloud/fog/snow haze.
    bright = np.clip((np.maximum(ya, yb) - 135.0) / 75.0, 0.0, 1.0)
    neutral = 1.0 - np.clip(np.minimum(sa, sb) / 65.0, 0.0, 1.0)

    # Avoid penalizing very bright sharp snow ridges too strongly: estimate
    # local gradient and reduce the cloud score on high-structure pixels.
    y = 0.5 * (ya + yb)
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    gy[1:-1, :] = 0.5 * (y[2:, :] - y[:-2, :])
    grad = np.hypot(gx, gy)
    low_structure = 1.0 - np.clip(grad / 38.0, 0.0, 1.0)

    return (bright * neutral * low_structure).astype(np.float32)


def _hard_diagonal_corner(region, diagonal, corner, overlap):
    """Resolve a true three-way grid junction with a full hard corner seam.

    For a diagonal overlap, the diagonal tile must own BOTH physical outer
    boundaries of the overlap×overlap corner. The current tile owns the inner
    side. A minimum-cost monotone seam runs across the full corner square.

    This is intentionally not a compact patch: at a three-way junction the
    entire outer edge can be visible in the final paste order.
    """
    if overlap <= 1:
        return {'extent_px': 0, 'seam_pixels': 0}

    target_corner = {
        'northwest': 'southeast',
        'northeast': 'southwest',
        'southwest': 'northeast',
        'southeast': 'northwest',
    }[corner]

    current = _corner_patch(region, corner, overlap)
    target = _corner_patch(diagonal, target_corner, overlap)

    # Both patches are world-oriented. Apply the SAME transform to both so the
    # physical outer corner of the current tile becomes canonical top-left.
    cur = _to_nw_corner(current, corner)
    tgt = _to_nw_corner(target, corner)

    cost = _corner_difference_cost(cur, tgt)
    extent = overlap - 1

    # Preserve the successful hard terrain seam, but discourage it from
    # crossing cloud/fog masses. This changes the PATH only; there is still
    # zero RGB blending and zero primary-overlap replacement.
    cloud_penalty = _cloud_presence_penalty(cur, tgt)
    cost = cost + 42.0 * cloud_penalty

    # The ideal junction runs from top-right to bottom-left. Content mismatch
    # may bend it, but a mild prior prevents pathological wandering.
    yy, xx = np.indices(cost.shape, dtype=np.float32)
    ideal_sum = float(extent)
    line_distance = np.abs((xx + yy) - ideal_sum)
    cost = cost + 0.10 * line_distance

    path = _minimum_corner_path(cost, extent)

    # Convert path into one boundary x per row. The diagonal tile owns the
    # canonical top-left side. Because the path spans the FULL corner:
    #   - full top edge is continuous with the diagonal tile,
    #   - full left edge is continuous with the diagonal tile,
    #   - current tile owns the inner/right/bottom side.
    boundary = np.full(overlap, -1, dtype=np.int32)
    for y, x in path:
        boundary[y] = max(boundary[y], int(x))

    # Fill any rows crossed only by a horizontal path segment.
    last = extent
    for y in range(overlap):
        if boundary[y] >= 0:
            last = boundary[y]
        else:
            boundary[y] = last

    out = cur.copy()
    for y in range(overlap):
        bx = int(np.clip(boundary[y], 0, overlap - 1))

        # Hard ownership: no RGB averaging, hence no ghosted terrain.
        # Include seam pixel in diagonal ownership so the two exposed outer
        # borders are pixel-identical to the already-published diagonal tile.
        out[y, :bx + 1] = tgt[y, :bx + 1]

    mixed = _from_nw_corner(out, corner)

    if corner == 'northwest':
        region[:overlap, :overlap] = mixed
    elif corner == 'northeast':
        region[:overlap, -overlap:] = mixed
    elif corner == 'southwest':
        region[-overlap:, :overlap] = mixed
    else:
        region[-overlap:, -overlap:] = mixed

    return {
        'extent_px': extent,
        'seam_pixels': len(path),
        'full_corner': True,
        'outer_edges_owned_by_diagonal': True,
    }



def reconcile_diagonal_corners(region, job, meta, layout):
    """Hard-cut against already-published diagonal cells that physically overlap."""
    jobs_root = Path(meta.get('jobs_root') or job.parent).resolve()
    regions = discover(jobs_root)
    tx = int(meta['target']['x'])
    ty = int(meta['target']['y'])
    ov = int(layout['overlap'])

    diagonals = {
        'northwest': (tx - 1, ty - 1),
        'northeast': (tx + 1, ty - 1),
        'southwest': (tx - 1, ty + 1),
        'southeast': (tx + 1, ty + 1),
    }

    reports = []
    for corner, pos in diagonals.items():
        diag_meta = regions.get(pos)
        if not diag_meta:
            continue

        diagonal = np.array(open_tile(diag_meta['image'], layout))
        seam = _hard_diagonal_corner(
            region, diagonal, corner, ov,
        )
        reports.append({
            'corner': corner,
            'diagonal_id': diag_meta['id'],
            'diagonal_position': list(pos),
            'overlap_px': ov,
            'seam_extent_px': seam['extent_px'],
            'seam_pixels': seam['seam_pixels'],
            'full_corner': seam['full_corner'],
            'outer_edges_owned_by_diagonal': seam['outer_edges_owned_by_diagonal'],
            'method': 'three_way_hard_corner_cloud_avoiding_v2',
        })

    return reports



def prepare_alignment(job, generated, expected, actual, meta, draft=False):
    with Image.open(job / 'input/generation_canvas.png') as im:
        original = np.array(im.convert('RGB'))
        resized_reference = np.array(im.convert('RGB').resize(actual, Image.Resampling.LANCZOS)
                                     .resize(expected, Image.Resampling.LANCZOS), dtype=np.float32)
    with Image.open(job / 'input/generation_mask.png') as im:
        fixed_image = Image.fromarray(np.where(np.array(im.convert('L')) == 0, 255, 0).astype(np.uint8))
    fixed = np.array(fixed_image.filter(ImageFilter.MinFilter(5))) > 0
    if not fixed.any():
        raise ValueError('No fixed anchor pixels in generation mask')

    error = float(np.abs(generated.astype(np.float32) - resized_reference)[fixed].mean())
    if error <= 12:
        return generated, {
            'method': 'identity',
            'anchor_mae_before': error,
            'anchor_mae_after': error,
            'status': 'accepted',
        }

    try:
        aligned, report = align_terrain(
            generated, original, fixed, meta['primary_side'],
            meta['core'], meta['overlap'], error
        )
        report['status'] = 'accepted'

        # Different alignment methods expose different diagnostic fields.
        if 'inliers' in report:
            print(
                f"Aligned terrain: {report['inliers']} landmarks; "
                f"MAE {error:.2f} -> {report['anchor_mae_after']:.2f}"
            )
        else:
            print(
                f"Aligned terrain with {report.get('method', 'fallback')}; "
                f"MAE {error:.2f} -> {report['anchor_mae_after']:.2f}"
            )
        return aligned, report

    except AlignmentError as exc:
        workflow = meta.get('workflow')

        # ChatGPT image generation may redraw the nominally fixed overlap enough
        # that reliable registration is impossible. In the ChatGPT tile workflow
        # we can still publish safely because stitching later reconstructs the
        # primary overlap from the real neighbor and blends it into the generated
        # core. Do not warp on unreliable evidence.
        if workflow in ('chatgpt_tile_canvas_v1', 'chatgpt_tile_canvas_v2'):
            report = {
                'method': 'unaligned_chatgpt_fallback',
                'status': 'accepted',
                'reason': str(exc),
                'anchor_mae_before': error,
                'anchor_mae_after': error,
                'registration_applied': False,
                'safe_publish_reason': (
                    'ChatGPT workflow restores/blends the primary overlap from '
                    'the existing neighbor instead of trusting the redrawn anchor.'
                ),
            }
            print(
                f'WARNING: alignment unavailable; continuing with safe ChatGPT '
                f'fallback: {exc}'
            )
            return generated, report

        report = {
            'method': 'unresolved',
            'status': 'rejected',
            'reason': str(exc),
            'anchor_mae_before': error,
            'anchor_mae_after': error,
        }
        if not draft:
            raise AlignmentError(
                f'Terrain could not be aligned: {exc}. '
                'Use --draft for inspection; the current tile is unchanged.'
            ) from exc
        print(f'WARNING: unaligned draft: {exc}')
        return generated, report


def stitch(job, generated=None, feather=4, draft=False, jobs='jobs'):
    if feather < 0:
        raise ValueError('feather must be non-negative')
    job, generated = resolve_job_and_generated(job, generated, jobs=jobs)
    if not generated.exists():
        raise FileNotFoundError(f'Generated image not found: {generated}')
    meta = load_json(job / 'grid_job.json')
    if meta.get('method') != 'local_tile_v1':
        raise ValueError('Legacy job: prepare a new local tile job first')
    layout = meta['layout']
    w, h, ov = layout['width'], layout['height'], layout['overlap']
    expected = tuple(meta['expected_generation_size'])
    with Image.open(generated) as im:
        actual = im.size
        sx, sy = actual[0] / expected[0], actual[1] / expected[1]
        if abs(sx - sy) / max(sx, sy) > 0.02:
            raise ValueError(f'Generated aspect ratio differs: expected {expected}, got {actual}')
        # Each stored tile uses the same world-space pixel scale.
        gen = np.array(im.convert('RGB').resize(expected, Image.Resampling.LANCZOS))
    if not meta.get('fixed_primary_overlap'):
        raise ValueError('Old job has no fixed overlap. Prepare new inputs and regenerate before stitching.')
    gen, alignment = prepare_alignment(job, gen, expected, actual, meta, draft=draft)
    anchor_error = alignment['anchor_mae_before']
    anchor_changed = anchor_error > 12.0
    side = meta['primary_side']
    ctx = meta['context']
    # Extract exactly one tile from the generated image.
    #
    # ChatGPT workflow always returns one complete tile. In that workflow
    # meta['context'] describes an external reference image only; it is NOT an
    # offset inside generated.png.
    #
    # Legacy/enlarged-canvas jobs still use context as a physical canvas offset.
    workflow = meta.get('workflow')
    if workflow in ('chatgpt_tile_canvas_v1', 'chatgpt_tile_canvas_v2'):
        if gen.shape[1] != w or gen.shape[0] != h:
            raise ValueError(
                f'ChatGPT tile workflow expected normalized tile {w}x{h}, '
                f'got {gen.shape[1]}x{gen.shape[0]}'
            )
        region = gen.copy()
    elif side == 'west':
        region = gen[:, ctx:ctx + w].copy()
    elif side == 'east':
        region = gen[:, :w].copy()
    elif side == 'north':
        region = gen[ctx:ctx + h, :].copy()
    else:  # south
        region = gen[:h, :].copy()

    if region.shape[:2] != (h, w):
        raise ValueError(
            f'Extracted tile has wrong size {region.shape[1]}x{region.shape[0]}; '
            f'expected {w}x{h}. workflow={workflow!r}, side={side!r}, context={ctx}'
        )
    # Primary first, then other cardinal neighbors. Later side merges may not
    # overwrite a corner that an earlier side already established.
    sides = [side] + [s for s in ('north', 'east', 'south', 'west')
                      if s != side and s in meta['neighbor_files']]
    side_merge_report = merge_sides_without_corner_overwrite(
        region, sides, meta['neighbor_files'], job, layout, feather
    )
    # The primary overlap was already resolved above with a HARD content-aware
    # seam. Do not run soften_primary_overlap() here: broad RGB blending is what
    # created ghosted houses, duplicated trees and blurred terrain when the two
    # generations were not pixel-identical.
    primary = np.array(open_tile(job / meta['neighbor_files'][side], layout))
    primary_overlap_restored = bool(
        not draft and alignment.get('status') == 'accepted'
    )
    overlap_transition = 0
    exact_boundary_band = 0

    # Diagonal grid cells overlap physically by ov×ov pixels even though they
    # are not returned by tile_grid.neighbors(). Reconcile those corner squares
    # explicitly after all side operations so one side cannot overwrite another.
    corner_reconciliation = reconcile_diagonal_corners(region, job, meta, layout)

    out = job / 'draft' if draft else job
    out.mkdir(exist_ok=True)
    output = out / 'tile.png'
    Image.fromarray(region).save(output)
    save_json(out / 'stitch_meta.json', {
        'format': 'local_tile_v1', 'generated_size': list(actual),
        'normalized_generation_size': list(expected), 'tile_size': [w, h],
        'workflow': meta.get('workflow'),
        'feather': feather, 'merged_sides': sides,
        'side_merge_report': side_merge_report,
        'anchor_mae': anchor_error,
        'anchor_changed': anchor_changed, 'draft': draft,
        'fixed_primary_overlap': primary_overlap_restored,
        'primary_overlap_restored': primary_overlap_restored,
        'primary_overlap_transition_px': overlap_transition,
        'exact_primary_boundary_px': exact_boundary_band,
        'primary_overlap_method': 'content_aware_hard_seam',
        'primary_overlap_broad_blend_disabled': True,
        'corner_reconciliation': corner_reconciliation,
        'alignment_method': alignment.get('method'),
        'registration_applied': alignment.get(
            'registration_applied',
            alignment.get('method') not in ('identity', 'unaligned_chatgpt_fallback')
        ),
        'alignment': alignment,
    })
    if draft:
        # Show this tile next to its primary neighbor; never register in the grid.
        if side in ('east', 'west'):
            pair = Image.new('RGB', (2 * w - ov, h))
            source_pos = (w - ov, 0) if side == 'east' else (0, 0)
            tile_pos = (0, 0) if side == 'east' else (w - ov, 0)
        else:
            pair = Image.new('RGB', (w, 2 * h - ov))
            source_pos = (0, h - ov) if side == 'south' else (0, 0)
            tile_pos = (0, 0) if side == 'south' else (0, h - ov)
        pair.paste(Image.fromarray(primary), source_pos)
        pair.paste(Image.fromarray(region), tile_pos)
        pair.thumbnail((1800, 1200), Image.Resampling.LANCZOS)
        pair.save(out / 'primary_pair_preview.png')
        print(out / 'primary_pair_preview.png')
        return
    # Dependencies must include every tile whose pixels were used. In
    # particular, diagonal reconciliation must become part of the dependency
    # graph or the simple final builder may paste that diagonal afterwards and
    # erase the corrected corner.
    dependencies = []
    seen_dependencies = set()

    for r in meta['neighbors'].values():
        dep = r['id']
        if dep not in seen_dependencies:
            dependencies.append(dep)
            seen_dependencies.add(dep)

    for report in corner_reconciliation:
        dep = report['diagonal_id']
        if dep not in seen_dependencies:
            dependencies.append(dep)
            seen_dependencies.add(dep)

    save_json(job / 'region.json', {
        'version': 1, 'format': 'local_tile_v1', 'id': meta['region_id'],
        **meta['target'], 'image': 'tile.png', 'layout': layout,
        'primary_side': side, 'primary_neighbor': meta['primary_neighbor']['id'],
        'dependencies': dependencies,
        'diagonal_dependencies': [
            r['diagonal_id'] for r in corner_reconciliation
        ],
    })
    print(output)


def main():
    p = argparse.ArgumentParser(description='Stitch one local tile; never write a cumulative world.')
    p.add_argument('--jobs', default='jobs',
                   help='Jobs root folder used when --job is a simple name (default: jobs)')
    p.add_argument('--job', required=True,
                   help='Job name like west_01 or a full path like jobs/west_01')
    p.add_argument('--generated', default=None,
                   help='Optional generated image path. Defaults to jobs/<job>/generated.png')
    p.add_argument('--feather', type=int, default=4)
    p.add_argument('--draft', action='store_true',
                   help='Inspect alignment (or an unaligned fallback); write only draft/, never publish a tile.')
    args = p.parse_args()
    stitch(args.job, args.generated, args.feather, draft=args.draft, jobs=args.jobs)


if __name__ == '__main__':
    main()
