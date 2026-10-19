#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from tile_grid import DIRS, geometry, discover, neighbors, open_tile, save_json


def resolve_job_path(jobs, job):
    jobs = Path(jobs).resolve()
    job = Path(job)
    if not job.is_absolute() and len(job.parts) == 1:
        job = jobs / job
    else:
        job = job.resolve()
    return jobs, job



LANDMARK_LIBRARY = [
    {
        'key': 'ruined_hill_fortress',
        'title': 'Ruined Hill Fortress',
        'summary': 'an old fortified hill with broken curtain walls and watchtowers',
        'features': [
            'stone curtain walls',
            'broken towers',
            'winding approach road',
            'terraced hill slopes',
            'small ruined outbuildings',
        ],
    },
    {
        'key': 'great_waterfall',
        'title': 'Great Waterfall',
        'summary': 'a dramatic waterfall or stepped cataract system',
        'features': [
            'waterfall cliffs',
            'mist plumes',
            'rocky gorge',
            'splitting streams',
            'spray-fed vegetation',
        ],
    },
    {
        'key': 'ancient_monastery',
        'title': 'Ancient Monastery',
        'summary': 'a secluded monastery or abbey complex',
        'features': [
            'stone cloisters',
            'walled gardens',
            'chapel or bell tower',
            'pilgrim path',
            'quiet cultivated terraces',
        ],
    },
    {
        'key': 'abandoned_mining_valley',
        'title': 'Abandoned Mining Valley',
        'summary': 'an old mining district with excavations and remnants of industry',
        'features': [
            'mine entrances',
            'tailing mounds',
            'timber scaffolds',
            'cart tracks',
            'small worker ruins',
        ],
    },
    {
        'key': 'sunken_ruins',
        'title': 'Sunken Ruins',
        'summary': 'partially flooded ancient ruins among marsh or shallow water',
        'features': [
            'half-submerged stone ruins',
            'causeways',
            'broken arches',
            'shallow branching waterways',
            'reed-filled pools',
        ],
    },
    {
        'key': 'giant_stone_bridge',
        'title': 'Giant Stone Bridge',
        'summary': 'a monumental ancient bridge or viaduct crossing a valley or river',
        'features': [
            'large stone arches',
            'approach roads',
            'deep ravine or river crossing',
            'eroded masonry',
            'support towers or gatehouses',
        ],
    },
    {
        'key': 'sacred_grove',
        'title': 'Sacred Grove',
        'summary': 'an unusually distinct sacred woodland with clearings and standing stones',
        'features': [
            'ancient trees',
            'forest clearings',
            'standing stones',
            'ritual paths',
            'small shrines',
        ],
    },
    {
        'key': 'battlefield_ruins',
        'title': 'Old Battlefield',
        'summary': 'an old battlefield scarred by earthworks, graves and ruined camps',
        'features': [
            'old embankments',
            'broken palisades',
            'grave mounds',
            'camp ruins',
            'scarred open ground',
        ],
    },
    {
        'key': 'volcanic_vent',
        'title': 'Volcanic Vent Field',
        'summary': 'a rugged geothermal or volcanic vent area',
        'features': [
            'blackened rock',
            'steam vents',
            'sulfur pools',
            'lava-like scarred ground',
            'sparse vegetation',
        ],
    },
    {
        'key': 'great_lake_shrine',
        'title': 'Lake Shrine',
        'summary': 'a shrine or temple complex on the shore of a striking lake',
        'features': [
            'broad lake edge',
            'causeway or pier',
            'stone shrine',
            'small islets',
            'reflective calm water',
        ],
    },
]


def _stable_hash(text):
    import hashlib
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)


def _frand(seed_text, idx=0):
    import random
    return random.Random(_stable_hash(f'{seed_text}|{idx}'))


def choose_landmark_cluster(x, y, seed='frontier_landmarks_v1', macro=4):
    """Return a smooth clustered landmark plan for tile (x, y).

    The map is partitioned into coarse macro cells. Each macro cell has a chance
    to host a landmark cluster center with jittered position, type and radius.
    Tiles inherit the strongest nearby cluster influence, which gives a smooth
    and somewhat unpredictable regional spread rather than a strict every-N rule.
    """
    gx0 = x // macro
    gy0 = y // macro
    best = None

    for gy in range(gy0 - 2, gy0 + 3):
        for gx in range(gx0 - 2, gx0 + 3):
            rng = _frand(f'{seed}:{gx}:{gy}')
            if rng.random() >= 0.36:
                continue

            cx = gx * macro + rng.uniform(0.35, macro - 0.35)
            cy = gy * macro + rng.uniform(0.35, macro - 0.35)
            radius = rng.uniform(1.45, 2.85)
            theme = LANDMARK_LIBRARY[rng.randrange(len(LANDMARK_LIBRARY))]
            dx = x - cx
            dy = y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > radius:
                continue

            influence = 1.0 - (dist / radius)
            influence = influence * influence * (3.0 - 2.0 * influence)

            if best is None or influence > best['influence']:
                best = {
                    'macro_cell': [gx, gy],
                    'center': [round(cx, 3), round(cy, 3)],
                    'radius_tiles': round(radius, 3),
                    'dist_tiles': round(dist, 3),
                    'influence': round(float(influence), 4),
                    'theme': theme,
                }

    if best is None or best['influence'] < 0.20:
        return {
            'enabled': False,
            'seed': seed,
            'macro_size': macro,
        }

    influence = best['influence']
    if influence >= 0.72:
        phase = 'core'
        emphasis = 'dominant but still geographically believable'
        intensity = 'strong'
    elif influence >= 0.44:
        phase = 'supporting'
        emphasis = 'clearly visible and region-defining, but not overwhelming'
        intensity = 'medium'
    else:
        phase = 'fringe'
        emphasis = 'subtle early signs that smoothly foreshadow the landmark cluster'
        intensity = 'light'

    return {
        'enabled': True,
        'seed': seed,
        'macro_size': macro,
        'phase': phase,
        'emphasis': emphasis,
        'intensity': intensity,
        **best,
    }




DIAGONAL_OFFSETS = {
    'northwest': (-1, -1),
    'northeast': (1, -1),
    'southwest': (-1, 1),
    'southeast': (1, 1),
}


def diagonal_neighbors(regions, x, y):
    """Return already-published diagonal cells around target (x, y)."""
    out = {}
    for corner, (dx, dy) in DIAGONAL_OFFSETS.items():
        r = regions.get((x + dx, y + dy))
        if r is not None:
            out[corner] = r
    return out


def apply_diagonal_overlap(canvas, mask, src, corner, ov):
    """Copy the exact physical overlap×overlap corner from a diagonal tile.

    The diagonal tile is offset in both axes, so it physically overlaps the new
    tile by exactly ov×ov pixels. It is applied AFTER cardinal strips and is
    authoritative for the three-way corner seen in the existing world.
    """
    h, w = canvas.shape[:2]

    if corner == 'northwest':
        # NW diagonal contributes its SE corner to target NW.
        canvas[:ov, :ov] = src[-ov:, -ov:]
        mask[:ov, :ov] = 0

    elif corner == 'northeast':
        # NE diagonal contributes its SW corner to target NE.
        canvas[:ov, w - ov:] = src[-ov:, :ov]
        mask[:ov, w - ov:] = 0

    elif corner == 'southwest':
        # SW diagonal contributes its NE corner to target SW.
        canvas[h - ov:, :ov] = src[:ov, -ov:]
        mask[h - ov:, :ov] = 0

    elif corner == 'southeast':
        # SE diagonal contributes its NW corner to target SE.
        canvas[h - ov:, w - ov:] = src[:ov, :ov]
        mask[h - ov:, w - ov:] = 0

    else:
        raise ValueError(f'Unknown diagonal corner: {corner}')


def diagonal_corner_crop(src, corner, ov, context):
    """Return a larger context crop around the diagonal source corner."""
    h, w = src.shape[:2]
    sx = min(w, ov + context)
    sy = min(h, ov + context)

    if corner == 'northwest':
        # Source is NW of target -> use its SE corner.
        return src[h - sy:, w - sx:]
    if corner == 'northeast':
        # Source is NE of target -> use its SW corner.
        return src[h - sy:, :sx]
    if corner == 'southwest':
        # Source is SW of target -> use its NE corner.
        return src[:sy, w - sx:]
    if corner == 'southeast':
        # Source is SE of target -> use its NW corner.
        return src[:sy, :sx]
    raise ValueError(f'Unknown diagonal corner: {corner}')


def extract_named_corner(src, named_corner, size):
    """Extract a square crop from a named corner of one tile."""
    h, w = src.shape[:2]
    s = min(size, h, w)
    if named_corner == 'nw':
        return src[:s, :s]
    if named_corner == 'ne':
        return src[:s, w - s:]
    if named_corner == 'sw':
        return src[h - s:, :s]
    if named_corner == 'se':
        return src[h - s:, w - s:]
    raise ValueError(f'Unknown named corner: {named_corner}')


def _feather_lr(left, right, blend=12):
    """Compose left/right crops with a narrow seam feather."""
    h = min(left.shape[0], right.shape[0])
    left = left[:h]
    right = right[:h]
    b = max(1, min(blend, left.shape[1] // 4, right.shape[1] // 4))
    out_w = left.shape[1] + right.shape[1] - b
    out = np.empty((h, out_w, 3), dtype=np.uint8)
    out[:, :left.shape[1] - b] = left[:, :left.shape[1] - b]
    out[:, left.shape[1]:] = right[:, b:]
    alpha = np.linspace(0.0, 1.0, b, dtype=np.float32)[None, :, None]
    seam = left[:, left.shape[1] - b:].astype(np.float32) * (1.0 - alpha) + right[:, :b].astype(np.float32) * alpha
    out[:, left.shape[1] - b:left.shape[1]] = np.clip(seam, 0, 255).astype(np.uint8)
    return out


def _feather_tb(top, bottom, blend=12):
    """Compose top/bottom crops with a narrow seam feather."""
    w = min(top.shape[1], bottom.shape[1])
    top = top[:, :w]
    bottom = bottom[:, :w]
    b = max(1, min(blend, top.shape[0] // 4, bottom.shape[0] // 4))
    out_h = top.shape[0] + bottom.shape[0] - b
    out = np.empty((out_h, w, 3), dtype=np.uint8)
    out[:top.shape[0] - b] = top[:top.shape[0] - b]
    out[top.shape[0]:] = bottom[b:]
    alpha = np.linspace(0.0, 1.0, b, dtype=np.float32)[:, None, None]
    seam = top[top.shape[0] - b:].astype(np.float32) * (1.0 - alpha) + bottom[:b].astype(np.float32) * alpha
    out[top.shape[0] - b:top.shape[0]] = np.clip(seam, 0, 255).astype(np.uint8)
    return out


def make_diagonal_bridge_guide(primary_src, primary_side, diagonal_src, diagonal_corner, ov, context, out_path):
    """Create a strong corner-junction reference showing how existing tiles already connect.

    This is NOT a preserve mask. It is a visual guide for the model:
    - if the primary side is north/south, compose the relevant diagonal tile crop
      beside the relevant primary crop (left/right seam already exists in the world);
    - if the primary side is east/west, compose them top/bottom.
    """
    size = min(primary_src.shape[0], primary_src.shape[1], diagonal_src.shape[0], diagonal_src.shape[1], ov + min(context, 320))

    if primary_side == 'south':
        if diagonal_corner == 'southwest':
            left = extract_named_corner(diagonal_src, 'ne', size)
            right = extract_named_corner(primary_src, 'nw', size)
            guide = _feather_lr(left, right)
            direction = 'The new tile lies ABOVE this guide, especially above the inner junction near the top-center.'
        elif diagonal_corner == 'southeast':
            left = extract_named_corner(primary_src, 'ne', size)
            right = extract_named_corner(diagonal_src, 'nw', size)
            guide = _feather_lr(left, right)
            direction = 'The new tile lies ABOVE this guide, especially above the inner junction near the top-center.'
        else:
            return None
    elif primary_side == 'north':
        if diagonal_corner == 'northwest':
            left = extract_named_corner(diagonal_src, 'se', size)
            right = extract_named_corner(primary_src, 'sw', size)
            guide = _feather_lr(left, right)
            direction = 'The new tile lies BELOW this guide, especially below the inner junction near the bottom-center.'
        elif diagonal_corner == 'northeast':
            left = extract_named_corner(primary_src, 'se', size)
            right = extract_named_corner(diagonal_src, 'sw', size)
            guide = _feather_lr(left, right)
            direction = 'The new tile lies BELOW this guide, especially below the inner junction near the bottom-center.'
        else:
            return None
    elif primary_side == 'east':
        if diagonal_corner == 'northeast':
            top = extract_named_corner(diagonal_src, 'sw', size)
            bottom = extract_named_corner(primary_src, 'nw', size)
            guide = _feather_tb(top, bottom)
            direction = 'The new tile lies LEFT of this guide, especially left of the inner junction near the center-left.'
        elif diagonal_corner == 'southeast':
            top = extract_named_corner(primary_src, 'sw', size)
            bottom = extract_named_corner(diagonal_src, 'nw', size)
            guide = _feather_tb(top, bottom)
            direction = 'The new tile lies LEFT of this guide, especially left of the inner junction near the center-left.'
        else:
            return None
    elif primary_side == 'west':
        if diagonal_corner == 'northwest':
            top = extract_named_corner(diagonal_src, 'se', size)
            bottom = extract_named_corner(primary_src, 'ne', size)
            guide = _feather_tb(top, bottom)
            direction = 'The new tile lies RIGHT of this guide, especially right of the inner junction near the center-right.'
        elif diagonal_corner == 'southwest':
            top = extract_named_corner(primary_src, 'se', size)
            bottom = extract_named_corner(diagonal_src, 'ne', size)
            guide = _feather_tb(top, bottom)
            direction = 'The new tile lies RIGHT of this guide, especially right of the inner junction near the center-right.'
        else:
            return None
    else:
        return None

    Image.fromarray(guide).save(out_path)
    return {
        'file': f'input/diagonals/{out_path.name}',
        'size': [int(guide.shape[1]), int(guide.shape[0])],
        'direction': direction,
        'corner': diagonal_corner,
        'primary_side': primary_side,
    }




def board(neigh, output):
    image = Image.new('RGB', (1260, 780), (18, 17, 15))
    draw = ImageDraw.Draw(image)
    positions = {'north': (420, 0), 'east': (840, 260),
                 'south': (420, 520), 'west': (0, 260)}
    draw.text((545, 380), 'NEW REGION', fill='white')
    for side, r in neigh.items():
        with Image.open(r['image']) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (420, 260))
        x, y = positions[side]
        image.paste(thumb, (x + (420 - thumb.width) // 2, y + (260 - thumb.height) // 2))
        draw.text((x + 8, y + 8), f"{side.upper()} {r['id']}", fill='white')
    image.save(output)


def save_side_context(src, side, ov, context, out_path):
    """Save overlap + extra context crop for one neighbor side."""
    h, w = src.shape[:2]
    if side == 'east':
        ctx = min(context, w - ov)
        crop = src[:, :ov + ctx]
    elif side == 'west':
        ctx = min(context, w - ov)
        crop = src[:, w - ov - ctx:]
    elif side == 'north':
        ctx = min(context, h - ov)
        crop = src[h - ov - ctx:, :]
    else:  # south
        ctx = min(context, h - ov)
        crop = src[:ov + ctx, :]
    Image.fromarray(crop).save(out_path)
    return int(ctx), crop.shape[1], crop.shape[0]


def apply_neighbor_overlap(canvas, mask, src, side, ov):
    """Copy one neighbor's exact shared overlap strip into the generation canvas."""
    h, w = canvas.shape[:2]
    if side == 'east':
        canvas[:, w - ov:] = src[:, :ov]
        mask[:, w - ov:] = 0
    elif side == 'west':
        canvas[:, :ov] = src[:, w - ov:]
        mask[:, :ov] = 0
    elif side == 'north':
        canvas[:ov] = src[-ov:]
        mask[:ov] = 0
    else:  # south
        canvas[h - ov:] = src[:ov]
        mask[h - ov:] = 0


def make_corner_guide(sources, sides, ov, context, out_path):
    """Create a compact corner reference when two perpendicular neighbors meet."""
    if len(sides) != 2:
        return None

    a, b = sides
    perp = {frozenset(['north', 'west']), frozenset(['north', 'east']),
            frozenset(['south', 'west']), frozenset(['south', 'east'])}
    if frozenset(sides) not in perp:
        return None

    sa = np.array(sources[a])
    sb = np.array(sources[b])
    h, w = sa.shape[:2]
    span_x = min(context + ov, w)
    span_y = min(context + ov, h)

    if frozenset(sides) == frozenset(['north', 'west']):
        crop_a = sa[h - span_y:, w - span_x:]   # from west tile, upper-right area
        crop_b = sb[h - span_y:, :span_x]       # from north tile, lower-left area
        corner_name = 'north_west'
    elif frozenset(sides) == frozenset(['north', 'east']):
        crop_a = sa[h - span_y:, :span_x]       # from east tile, upper-left area
        crop_b = sb[h - span_y:, w - span_x:]   # from north tile, lower-right area
        corner_name = 'north_east'
    elif frozenset(sides) == frozenset(['south', 'west']):
        crop_a = sa[:span_y, w - span_x:]       # from west tile, lower-right area
        crop_b = sb[:span_y, :span_x]           # from south tile, upper-left area
        corner_name = 'south_west'
    else:
        crop_a = sa[:span_y, :span_x]           # from east tile, lower-left area
        crop_b = sb[:span_y, w - span_x:]       # from south tile, upper-right area
        corner_name = 'south_east'

    pad = 18
    title_h = 28
    panel_w = crop_a.shape[1]
    panel_h = crop_a.shape[0]
    guide = Image.new('RGB', (panel_w * 2 + pad * 3, panel_h + title_h + pad * 2), (18, 17, 15))
    draw = ImageDraw.Draw(guide)
    draw.text((pad, 6), f'CORNER GUIDE: {corner_name.upper()}', fill='white')
    draw.text((pad, title_h), a.upper(), fill='white')
    draw.text((pad * 2 + panel_w, title_h), b.upper(), fill='white')
    guide.paste(Image.fromarray(crop_a), (pad, title_h + pad))
    guide.paste(Image.fromarray(crop_b), (pad * 2 + panel_w, title_h + pad))
    guide.save(out_path)
    return {
        'file': str(out_path.name),
        'corner': corner_name,
        'sides': [a, b],
        'size': [guide.width, guide.height],
    }


def prepare(jobs, job, x, y, primary_side, region_id=None, overlap=None,
            context=700, width=None, height=None):
    jobs, job = resolve_job_path(jobs, job)
    if job.parent != jobs or job.name.startswith('_') or job.name == 'center':
        raise ValueError('Job must be a direct child of jobs, excluding center and _*')
    if job.exists() and any(job.iterdir()):
        raise FileExistsError(f'{job} is not empty; use a new job directory')

    layout = geometry(jobs, width, height, overlap)
    regs = discover(jobs)
    if (x, y) in regs:
        raise ValueError(f'Grid position {(x, y)} is occupied')

    region_id = region_id or job.name
    if region_id in {r['id'] for r in regs.values()}:
        raise ValueError(f'Region ID already exists: {region_id}')

    neigh = neighbors(regs, x, y)
    if primary_side not in neigh:
        raise ValueError(f'Primary {primary_side} missing; available: {list(neigh)}')

    diag = diagonal_neighbors(regs, x, y)

    sources = {s: open_tile(r['image'], layout) for s, r in neigh.items()}
    diagonal_sources = {
        corner: open_tile(r['image'], layout)
        for corner, r in diag.items()
    }
    w, h, ov = layout['width'], layout['height'], layout['overlap']

    canvas = np.zeros((h, w, 3), np.uint8)
    mask = np.full((h, w), 255, np.uint8)

    # Fix ALL known neighbor overlaps, not only the primary one. This gives the
    # generator a strong L/T-shaped anchor when two or more neighbors exist.
    apply_order = [primary_side] + [s for s in neigh if s != primary_side]
    for side in apply_order:
        apply_neighbor_overlap(canvas, mask, np.array(sources[side]), side, ov)

    # Diagonal neighbors are REFERENCE ONLY.
    # Do NOT write their overlap×overlap pixels into generation_canvas/mask.
    # Freezing the diagonal square creates an over-constrained three-way corner
    # and tends to produce pasted/floating mountain masses. Cardinal overlaps
    # remain exact black/preserve anchors; diagonals guide composition only.

    # The target core is the region not occupied by the primary overlap strip.
    core = (w - ov) if primary_side in ('east', 'west') else (h - ov)

    inp = job / 'input'
    neighbors_dir = inp / 'neighbors'
    diagonals_dir = inp / 'diagonals'
    neighbors_dir.mkdir(parents=True)
    diagonals_dir.mkdir(parents=True)

    Image.fromarray(canvas).save(inp / 'generation_canvas.png')
    Image.fromarray(mask).save(inp / 'generation_mask.png')
    sources[primary_side].save(inp / 'source_region_native.png')

    neighbor_files = {primary_side: 'input/source_region_native.png'}
    context_files = {}
    corner_guides = []

    for side, image in sources.items():
        if side != primary_side:
            file_rel = f'input/neighbors/{side}.png'
            image.save(job / file_rel)
            neighbor_files[side] = file_rel

        ctx_file = inp / ('primary_context.png' if side == primary_side else f'neighbors/{side}_context.png')
        actual_ctx, ctx_w, ctx_h = save_side_context(np.array(image), side, ov, context, ctx_file)
        rel_ctx = 'input/primary_context.png' if side == primary_side else f'input/neighbors/{side}_context.png'
        context_files[side] = {
            'file': rel_ctx,
            'context_px': actual_ctx,
            'size': [ctx_w, ctx_h],
        }

    diagonal_files = {}
    diagonal_contexts = {}
    diagonal_bridge_guides = {}

    for corner, image in diagonal_sources.items():
        full_rel = f'input/diagonals/{corner}.png'
        image.save(job / full_rel)
        diagonal_files[corner] = full_rel

        crop = diagonal_corner_crop(
            np.array(image),
            corner,
            ov,
            min(context, min(w, h) - ov),
        )
        ctx_rel = f'input/diagonals/{corner}_corner_context.png'
        Image.fromarray(crop).save(job / ctx_rel)
        diagonal_contexts[corner] = {
            'file': ctx_rel,
            'size': [int(crop.shape[1]), int(crop.shape[0])],
            'overlap_px': ov,
            'neighbor_id': diag[corner]['id'],
        }

        guide = make_diagonal_bridge_guide(
            np.array(sources[primary_side]),
            primary_side,
            np.array(image),
            corner,
            ov,
            context,
            diagonals_dir / f'{corner}_bridge_guide.png',
        )
        if guide:
            diagonal_bridge_guides[corner] = guide

    # Add compact corner guides for every perpendicular secondary neighbor.
    for side in [s for s in neigh if s != primary_side]:
        if ({primary_side, side} <= {'north', 'south'}) or ({primary_side, side} <= {'east', 'west'}):
            continue
        guide = make_corner_guide(
            sources,
            [primary_side, side],
            ov,
            min(context, min(w, h) - ov),
            inp / f'corner_guide_{primary_side}_{side}.png'
        )
        if guide:
            guide['file'] = f'input/{guide["file"]}'
            corner_guides.append(guide)

    board(neigh, inp / 'context_board.png')

    fixed_sides = apply_order
    secondary_sides = [s for s in neigh if s != primary_side]
    corner_sides = [g['sides'][1] for g in corner_guides]

    landmark_plan = choose_landmark_cluster(x, y)

    prompt_lines = [
        'Create one continuous regional map expansion for The Frontier.',
        f'Return exactly one new tile image sized {w}x{h}. Do NOT return the larger world,',
        'do NOT add blank margins, and do NOT change the aspect ratio.',
        f'Primary existing neighbor: {primary_side.upper()}. Shared overlap: {ov}px.',
        f'Known neighbors around the new tile: ' + ', '.join(f'{s.upper()}={neigh[s]["id"]}' for s in apply_order) + '.',
        'Use generation_canvas.png and generation_mask.png: white means generate,',
        'black means preserve exactly. Every black strip already comes from a real existing neighbor.',
        'Do not redraw, shift, scale or reinterpret any black-masked pixels.',
        f'Fixed overlap sides in the canvas: {", ".join(s.upper() for s in fixed_sides)}.',
        '',
        'Additional references:',
        f'- source_region_native.png is the full primary neighbor tile ({primary_side.upper()}).',
        f'- the {primary_side.upper()} side is the main terrain authority: inherit its large landforms first, then introduce new variation deeper inside the tile.',
        f'- primary_context.png shows extra context beyond the primary overlap ({context_files[primary_side]["context_px"]}px).',
        f'- where the {primary_side.upper()} border approaches a diagonal-corner reference, keep the primary overlap exact and generate a gradual bridge toward the diagonal terrain mass and cloud flow.',
    ]

    for side in secondary_sides:
        prompt_lines.append(
            f'- input/neighbors/{side}.png is the full {side.upper()} neighbor tile; '
            f'input/neighbors/{side}_context.png shows extra context beyond that overlap '
            f'({context_files[side]["context_px"]}px).'
        )

    if corner_guides:
        prompt_lines.append(
            '- corner_guide_*.png images show the critical corner where the primary overlap meets a secondary neighbor.'
        )
    if diag:
        prompt_lines.extend([
            '',
            'Critical diagonal corner references:',
            '- diagonal tiles are visual/compositional references only; they are NOT black/preserve pixels in generation_canvas.png.',
            '- use them to continue corner-specific terrain masses and cloud/fog flow, but do not paste or rigidly reproduce the diagonal tile.',
            '- keep the primary/cardinal black overlap exact.',
            '',
            'Corner bridge rule:',
            '- the model must create a natural bridge between the exact primary overlap and the diagonal reference.',
            '- the first 180–320 px inward from the relevant corner should favor continuation over novelty.',
            '- do not create a floating new plain, abrupt cliff break, pasted mountain block, or disconnected cloud bank beside the corner.',
            '- if the diagonal shows mountains and clouds near the corner, continue those masses/directions smoothly into the new tile while still honoring the primary overlap.',
        ])
        for corner, r in diag.items():
            prompt_lines.append(
                f'- {corner.upper()}: {r["id"]}; full tile=input/diagonals/{corner}.png; '
                f'corner context=input/diagonals/{corner}_corner_context.png.'
            )
            if corner in diagonal_bridge_guides:
                prompt_lines.append(
                    f'- {corner.upper()} bridge guide: {diagonal_bridge_guides[corner]["file"]}. '
                    f'{diagonal_bridge_guides[corner]["direction"]}'
                )

    prompt_lines.extend([
        '',
        'Highest priority: the new tile must continue smoothly from EVERY existing neighbor,',
        'not only from the primary one. Respect all fixed overlap strips simultaneously.',
        'Large-scale terrain continuity is mandatory, not optional.',
        '- treat preserved borders as the start of real landforms, not just color references.',
        '- if a preserved border contains a mountain wall, ridge chain, cliff band, steep slope or foothill system, continue that terrain mass naturally into the white area.',
        '- do NOT let mountains abruptly stop at the border and immediately switch to a wide new flat plain, open basin or unrelated terrain regime.',
        '- mountain masses should continue, branch, step down through foothills, open into passes, or gradually descend into valleys.',
        '- rivers, lakes and coastlines must respect the existing relief; do not create floating-looking terrain shelves.',
        '- keep the perceived elevation structure coherent across the border: ridge direction, valley depth, foothills and foreground scale must stay believable.',
        '- clouds or fog may continue across the border, but they must not be used to hide a broken terrain transition.',
    ])

    if corner_sides:
        prompt_lines.append(
            'Pay special attention to the shared corner where ' +
            ' and '.join(f'{primary_side.upper()} + {side.upper()}' for side in corner_sides) +
            ' meet: terrain, rivers, roads, forests and settlements must remain coherent through that corner.'
        )

    if diag:
        prompt_lines.append(
            'For every diagonal corner reference, preserve terrain logic and cloud flow continuity: '
            'continue the visible ridge direction, foothill structure, forest masses and cloud banks into the new territory without literal copy-pasting.'
        )
        prompt_lines.append(
            'Use any *_bridge_guide.png as the strongest clue for the troublesome corner transition: '
            'it shows the already-existing junction around the target, and the new tile should feel like a believable continuation away from that junction.'
        )

    if landmark_plan.get('enabled'):
        theme = landmark_plan['theme']
        prompt_lines.extend([
            '',
            'Special landmark plan:',
            f"- cluster: {theme['title']}",
            f"- phase in this tile: {landmark_plan['phase']}",
            f"- influence strength: {landmark_plan['influence']} ({landmark_plan['intensity']})",
            f"- summary: {theme['summary']}",
            f"- desired emphasis: {landmark_plan['emphasis']}",
            f"- suggested features: {', '.join(theme['features'])}",
            '- rule: preserve all neighbor continuity first, then let the new territory gradually reveal this landmark theme, strongest away from fixed overlaps.',
            '- use only 2-3 dominant supporting cues; do not force every listed feature into the image.',
        ])
    else:
        prompt_lines.extend([
            '',
            'Special landmark plan:',
            '- cluster: none for this tile',
            '- rule: avoid introducing a major landmark; keep this tile as a transition or standard region.',
        ])

    prompt_lines.extend([
        'Continue rivers, roads, ridges, coastlines, forests and settlements naturally into the new territory.',
        'Preserve camera, terrain scale, lighting, river width, forest density and overall style.',
        'High oblique 3/4 aerial regional map, realistic painterly medieval frontier.',
        'No UI, text, labels, borders, markers, mirrored copies or panel divisions.',
    ])

    (inp / 'prompt.txt').write_text('\n'.join(prompt_lines) + '\n', encoding='utf-8')

    save_json(job / 'grid_job.json', {
        'version': 1,
        'method': 'local_tile_v1',
        'region_id': region_id,
        'jobs_root': str(jobs),
        'target': {'x': x, 'y': y},
        'layout': layout,
        'primary_side': primary_side,
        'primary_neighbor': neigh[primary_side],
        'neighbors': neigh,
        'neighbor_files': neighbor_files,
        'secondary_neighbors': {s: r for s, r in neigh.items() if s != primary_side},
        'target_region_size': [w, h],
        'expected_generation_size': [w, h],
        'core': core,
        'overlap': ov,
        'context': context_files[primary_side]['context_px'],
        'neighbor_contexts': context_files,
        'corner_guides': corner_guides,
        'diagonal_neighbors': diag,
        'diagonal_files': diagonal_files,
        'diagonal_contexts': diagonal_contexts,
        'diagonal_bridge_guides': diagonal_bridge_guides,
        'fixed_diagonal_corners': [],
        'diagonal_reference_corners': sorted(diag.keys()),
        'diagonal_anchor_version': 5,
        'diagonal_anchor_mode': 'reference_only_soft_corner_bridge',
        'terrain_continuity_version': 2,
        'corner_influence_zone_px': [180, 280],
        'landmark_plan': landmark_plan,
        'fixed_sides': fixed_sides,
        'fixed_primary_overlap': True,
        # Keep v2 for compatibility with the existing stitch script. The extra
        # diagonal-anchor fields are additive metadata.
        'workflow': 'chatgpt_tile_canvas_v2',
    })
    save_json(jobs / 'layout.json', layout)
    print(
        f'Target ({x},{y}); tile={w}x{h}; fixed_sides={fixed_sides}; '
        f'diagonal_references={sorted(diag.keys())}; terrain_continuity=on'
    )
    print(inp)


def main():
    p = argparse.ArgumentParser(description='Prepare a ChatGPT-friendly local tile job with strong multi-neighbor anchoring.')
    p.add_argument('--jobs', default='jobs')
    p.add_argument('--job', required=True,
                   help='Job name like west_01 or a full path like jobs/west_01')
    p.add_argument('--x', type=int, required=True)
    p.add_argument('--y', type=int, required=True)
    p.add_argument('--primary', choices=DIRS, required=True)
    p.add_argument('--id')
    p.add_argument('--overlap', type=int, default=None)
    p.add_argument('--context', type=int, default=700)
    p.add_argument('--cell-width', type=int, default=None)
    p.add_argument('--cell-height', type=int, default=None)
    args = p.parse_args()
    prepare(Path(args.jobs), args.job, args.x, args.y, args.primary,
            args.id, args.overlap, args.context, args.cell_width, args.cell_height)


if __name__ == '__main__':
    main()
