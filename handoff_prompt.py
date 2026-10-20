"""A self-contained image-edit instruction derived from the actual attached mask.

Only canvas coordinates belong in this prompt. World/tile coordinates remain
in zones.json for compositing and never compete with the edit instructions.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from tile_grid import load_json

PROMPT_VERSION = 2


def rectangles(pixels):
    """Exact disjoint half-open rectangles for a boolean raster, without hulls."""
    pixels = np.asarray(pixels, dtype=bool)
    active, result = {}, []
    for y, row in enumerate(pixels):
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        spans = set(zip(np.flatnonzero(edges == 1).tolist(),
                        np.flatnonzero(edges == -1).tolist()))
        for span in list(active):
            if span not in spans:
                result.append(active.pop(span))
        for left, right in sorted(spans):
            if (left, right) in active:
                active[left, right][3] = y + 1
            else:
                active[left, right] = [left, y, right, y + 1]
    result.extend(active.values())
    return sorted(result, key=lambda r: (r[1], r[0], r[3], r[2]))


def _describe(rect):
    a, b, c, d = rect
    return f'x={a} through {c-1}, y={b} through {d-1} (inclusive)'


def render_prompt(meta, zones, mask, labels):
    mask, labels = np.asarray(mask), np.asarray(labels)
    if mask.ndim != 2 or labels.shape != mask.shape:
        raise ValueError('Prompt mask/zone dimensions differ')
    h, w = mask.shape
    if [w, h] != list(zones['canvas_size']):
        raise ValueError('Prompt canvas dimensions differ from zones')
    if not np.all((mask == 0) | (mask == 255)):
        raise ValueError('Prompt requires a binary generation mask')
    if not np.array_equal(mask == 255, (labels == 2) | (labels == 3)):
        raise ValueError('Prompt edit mask differs from generation zones')
    edits = rectangles(mask == 255)
    if not edits:
        raise ValueError('No editable pixels in current handoff')
    padding = rectangles((mask == 0) & (labels == 0))
    context = rectangles(labels == 1)
    identifier = hashlib.sha256(mask.tobytes()).hexdigest()[:12]
    lines = [
        'Use case: compositing. Perform a PRECISE LOCAL OUTPAINT EDIT of generation_canvas.png.',
        'generation_canvas.png is the EDIT TARGET. generation_mask.png is ONLY the mask reference, never the output image.',
        'Use these two currently attached files only. Ignore any earlier canvases or failed generated images.',
        f'Return the COMPLETE canvas at EXACTLY {w}x{h} pixels as generated.png.',
        'Keep the same framing, native pixel scale, camera and placement of every existing feature.',
        f'Pass {zones["pass"]}/{zones["pass_count"]}. Mask ID {identifier}. Do not draw this ID or any text.',
        '',
        f'ALL COORDINATES BELOW REFER ONLY TO THE ATTACHED {w}x{h} CANVAS.',
        'Origin is its top-left pixel; x increases rightward, y downward. Ranges are inclusive.',
        'EDIT ONLY these WHITE-mask rectangles:',
    ]
    lines.extend(f'- {_describe(r)}.' for r in edits)
    lines.extend([
        'Every pixel OUTSIDE those rectangles is BLACK in the mask and must remain unchanged.',
        'Do NOT enlarge, extend, soften or reinterpret the edit mask.',
        'The black pixels in the CANVAS inside a white-mask rectangle are missing terrain to fill.',
        'Other plain-black areas of the CANVAS are padding to preserve, not additional outpainting space.',
        '',
        'LOCKED EXISTING TERRAIN: keep every feature at its original location and scale:',
    ])
    lines.extend(f'- Preserve {_describe(r)}.' for r in context)
    if padding:
        lines.append('LOCKED PLAIN-BLACK PADDING: leave these rectangles plain black:')
        lines.extend(f'- {_describe(r)}.' for r in padding)
    lines.extend([
        '',
        'Grow the existing scene into the editable area; do not compose an independent landscape and attach it.',
        'First connect the local landforms at the actual edit boundary: slope to slope, valley to valley, water to water.',
        'If a river or lake meets the editable boundary, continue THAT watercourse and shoreline without a break.',
        'Continue the actual visible forest contours, paths and ridges at their current elevation and direction.',
        'Do not close existing water or a valley with a new mountain wall, cliff shelf or straight forest wall.',
        'The rectangular edit boundary must never appear as a geological division, vertical seam or separator line.',
        'Do not move, resize, rotate, mirror, redraw or reframe the locked lake, castle, waterfall, mountains or settlements.',
        'Do not duplicate unique landmarks into new territory. Match the reference lighting, scale and painterly aerial style.',
        'Keep both corner connections natural. No interface, labels, borders, frames or divider lines.',
        f'Return the full {w}x{h} PNG including every locked padding rectangle; do not crop or trim black bars.',
        'Do not output the mask, a comparison board, a screenshot or a newly framed full-scene illustration.',
    ])
    return '\n'.join(lines) + '\n'


def refresh_prompt(job):
    """Update only the text handoff; retain generation, images and pass progress."""
    job = Path(job)
    meta = load_json(job / 'grid_job.json')
    if meta.get('prepare_version') != 3:
        raise ValueError('refresh-prompt requires a native-patch job; rebuild legacy jobs first')
    inp = job / 'input'
    zones = load_json(inp / 'zones.json')
    with Image.open(inp / 'generation_mask.png') as im:
        mask = np.array(im.convert('L'))
    with Image.open(inp / 'generation_zones.png') as im:
        labels = np.array(im.convert('L'))
    with Image.open(inp / 'generation_canvas.png') as im:
        if list(im.size) != zones['canvas_size']:
            raise ValueError('Canvas and prompt geometry differ')
    text = render_prompt(meta, zones, mask, labels)
    path = inp / 'prompt.txt'
    if path.is_file() and path.read_text() == text:
        return {'job': job.name, 'prompt': str(path), 'changed': False,
                'pass': zones['pass'], 'pass_count': zones['pass_count']}
    backup = None
    if path.is_file():
        revision = inp / 'prompt_revisions'
        revision.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = revision / (stamp + '.txt')
        backup.write_bytes(path.read_bytes())
    temp = path.with_suffix('.txt.tmp')
    try:
        temp.write_text(text)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return {'job': job.name, 'prompt': str(path), 'changed': True,
            'backup': str(backup) if backup else None,
            'pass': zones['pass'], 'pass_count': zones['pass_count'],
            'next_step': 'Use the unchanged current canvas and mask with this updated prompt. Existing pass progress and generated.png are preserved.'}
