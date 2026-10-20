"""Controlled overlap handoff: generate into part of the shared strip, then cut
at a terrain-aware path. No world-side images are modified and no RGB crossfade
can create ghost trees or buildings.
"""
import numpy as np
from seam_refinement import tone_match_generated


def _orient(array, side):
    if side == 'east':
        return array
    if side == 'west':
        return np.flip(array, axis=1)
    if side == 'north':
        return np.rot90(array, 3)
    if side == 'south':
        return np.rot90(array, 1)
    raise ValueError(f'Unknown primary side {side}')


def _unorient(array, side):
    if side == 'east':
        return array
    if side == 'west':
        return np.flip(array, axis=1)
    if side == 'north':
        return np.rot90(array, 1)
    if side == 'south':
        return np.rot90(array, 3)
    raise ValueError(f'Unknown primary side {side}')


def editable_overlap_mask(lock_mask, side, overlap, *, min_fixed=64, corner_pad=24):
    """Expose an interior part of a cardinal overlap for generation.

    Far/outer pixels remain black in the model mask and are exact world
    pixels. Full diagonal-corner regions are also kept black.
    The original full mask remains separately stored as neighbor_mask.png.
    """
    oriented = _orient(np.array(lock_mask, copy=True), side)
    height, width = oriented.shape
    ov = int(overlap)
    guard = min(ov, max(8, min(int(min_fixed), ov // 4)))
    pad = min(int(corner_pad), max(0, (height - 2 * ov) // 6))
    y0, y1 = ov + pad, height - ov - pad
    x0, x1 = width - ov, width - guard
    if x1 <= x0 or y1 <= y0:
        return np.array(lock_mask, copy=True), None
    # Only release pixels already occupied by other tiles, never enlarge the
    # physical overlap into an unconnected area.
    original = oriented[y0:y1, x0:x1]
    if not np.all(original == 0):
        # Cardinal corners / irregular masks may interrupt this rectangle;
        # stay conservative rather than open an unsafe hole.
        return np.array(lock_mask, copy=True), None
    oriented[y0:y1, x0:x1] = 255
    return np.ascontiguousarray(_unorient(oriented, side)), {
        'side': side,
        'editable_strip_width': x1 - x0,
        'fixed_outer_width': guard,
        'top_bottom_protection': [y0, height - y1],
    }


def _seam_path(cost):
    """Monotone seam with endpoints pinned at the original internal edge."""
    height, width = cost.shape
    dp = np.full(width, np.inf, dtype=np.float64)
    dp[0] = float(cost[0, 0])
    parents = np.zeros((height, width), dtype=np.int32)
    xx = np.arange(width)
    for y in range(1, height):
        left = np.empty(width, dtype=np.float64)
        right = np.empty(width, dtype=np.float64)
        left[0] = np.inf
        left[1:] = dp[:-1]
        right[-1] = np.inf
        right[:-1] = dp[1:]
        all_previous = np.stack((left, dp, right))
        chosen = np.argmin(all_previous, axis=0)
        parents[y] = np.clip(xx + chosen - 1, 0, width - 1)
        dp = cost[y] + all_previous[chosen, xx]
    path = np.zeros(height, dtype=np.int32)
    # Force the path back to the original internal seam at the other corner.
    path[-1] = 0
    for y in range(height - 1, 0, -1):
        path[y - 1] = parents[y, path[y]]
    return path


def merge_editable_overlap(generated, locked, lock_mask, generated_mask, side):
    """Return exact locked world with a terrain-aware path in the editable band.

    Where the model mask is BLACK, all pixels are restored exactly. On the
    editable band, one side of an adaptive HARD seam is generated art, and
    the other side remains the old world. A failure to identify a valid band
    raises rather than breaking terrain connectivity silently.
    """
    if generated.shape != locked.shape or lock_mask.shape != generated.shape[:2]:
        raise ValueError('Invalid overlap arrays')
    if generated_mask.shape != lock_mask.shape:
        raise ValueError('Generation mask differs from neighbor lock')
    if np.any((generated_mask == 0) & (lock_mask != 0)):
        raise ValueError('Generation mask attempts to lock unknown pixels')

    candidate = np.array(generated, copy=True)
    candidate[lock_mask == 0] = locked[lock_mask == 0]
    editable = (lock_mask == 0) & (generated_mask != 0)
    if not np.any(editable):
        return candidate, {'method': 'exact_lock', 'seam_pixels': 0}

    g = np.array(_orient(generated, side), copy=True)
    l = _orient(locked, side)
    active = _orient(editable, side)
    ys, xs = np.nonzero(active)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if not np.all(active[y0:y1, x0:x1]) or int(active.sum()) != (y1 - y0) * (x1 - x0):
        raise ValueError('Editable overlap must be a single rectangle')
    height, width = y1 - y0, x1 - x0
    if width < 2 or height < 3:
        return candidate, {'method': 'exact_lock', 'seam_pixels': 0}

    # 96px generated-only tonal ramp; never blend world geometry or recolor
    # existing neighbors. Adaptive cut is selected AFTER the tone correction.
    g, tonal = tone_match_generated(g, l, x0, x1, y0, y1, radius=96)
    out = np.array(g, copy=True)
    known = _orient(lock_mask == 0, side)
    out[known] = l[known]
    # A hard seam follows low color/structure disagreement; the endpoints
    # stay anchored where the existing core met the original locked strip.
    a = g[y0:y1, x0:x1].astype(np.float32)
    b = l[y0:y1, x0:x1].astype(np.float32)
    mismatch = np.abs(a - b).mean(axis=2)
    xx = np.arange(width, dtype=np.float32)[None, :]
    # Discourage a cut into the far outer border and arbitrary excursions.
    cost = mismatch + 1.5 * (xx / max(1, width - 1)) ** 2
    seam = _seam_path(cost)
    for row, boundary in enumerate(seam):
        if boundary:
            out[y0 + row, x0:x0 + int(boundary)] = g[y0 + row, x0:x0 + int(boundary)]
    # Preserve every black pixel exactly, including diagonal corners.
    preserve = _orient(generated_mask == 0, side)
    out[preserve] = l[preserve]
    return np.ascontiguousarray(_unorient(out, side)), {
        'method': 'terrain_aware_hard_cut',
        'tonal_match': tonal,
        'seam_path_px': [int(x0 + v) for v in seam],
        'seam_y_start': int(y0),
        'seam_pixels': int(height),
        'seam_x_min': int(x0 + seam.min()),
        'seam_x_max': int(x0 + seam.max()),
        'seam_x_mean': float(x0 + seam.mean()),
        'band_width': int(width),
        'max_rgb_disagreement_at_cut': float(mismatch[np.arange(height), seam].max()),
        'avg_rgb_disagreement_at_cut': float(mismatch[np.arange(height), seam].mean()),
    }
