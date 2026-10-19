"""OpenAI Images Edit adapter for the Frontier local-tile generation pipeline.

This adapter NEVER scales a canvas or a generated image. It only adds a few
pixels of padding (when needed), removes the exact padding, and restores each
masked-off source pixel before atomically saving the generated PNG.

Frontier mask: 0 (black) = fixed, 255 (white) = generate.
OpenAI mask alpha: 255 = preserve hint, 0 = edit hint.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

from PIL import Image

MODEL_ALLOWLIST = {'gpt-image-2.5-sunburst', 'gpt-image-2.5-flare'}
QUALITY_ALLOWLIST = {'low', 'medium', 'high', 'xhigh', 'max', 'auto'}


def get_dimensions(w: int, h: int) -> tuple[int, int]:
    """Pad right/bottom up to multiples of 16; refuse unsupported API sizes."""
    if w <= 0 or h <= 0:
        raise ValueError(f'Invalid original canvas size: {w}x{h}')
    pw, ph = ((w + 15) // 16) * 16, ((h + 15) // 16) * 16
    if max(pw, ph) > 3840 or not (655_360 <= pw * ph <= 8_294_400) or not (1 / 3 <= pw / ph <= 3):
        raise ValueError(
            f'Canvas {w}x{h} requires {pw}x{ph}, outside the supported '
            'GPT Image 2.5 custom-size range; refusing to resize it.'
        )
    return pw, ph


def create_padded_inputs(canvas: Image.Image, frontier_mask: Image.Image):
    """Make the same-sized RGBA PNG input and alpha PNG mask, without resize."""
    canvas = canvas.convert('RGBA')
    frontier_mask = frontier_mask.convert('L')
    if canvas.size != frontier_mask.size:
        raise ValueError('Canvas and Frontier mask sizes differ')
    w, h = canvas.size
    pw, ph = get_dimensions(w, h)

    # Extend the reference with literal border pixels (copy only, no interpolation)
    # so that the few extra columns/rows do not form a transparent black seam.
    padded = Image.new('RGBA', (pw, ph))
    padded.paste(canvas, (0, 0))
    if pw > w:
        right_column = canvas.crop((w - 1, 0, w, h))
        for x in range(w, pw):
            padded.paste(right_column, (x, 0))
    if ph > h:
        bottom_row = padded.crop((0, h - 1, pw, h))
        for y in range(h, ph):
            padded.paste(bottom_row, (0, y))

    # Padded area is marked editable; it is removed after the API response.
    # Preserve only exact mask=0 pixels, matching the stitcher's constraints.
    padded_mask = Image.new('L', (pw, ph), 255)
    padded_mask.paste(frontier_mask, (0, 0))
    alpha = padded_mask.point(lambda p: 255 if p == 0 else 0)
    openai_mask = Image.new('RGBA', (pw, ph), (255, 255, 255, 255))
    openai_mask.putalpha(alpha)
    return padded, openai_mask


def restore_original_fixed_pixels(original: Image.Image,
                                  generated_padded: Image.Image,
                                  frontier_mask: Image.Image) -> Image.Image:
    """Reject model size drift; crop padding; preserve fixed pixels exactly."""
    original = original.convert('RGBA')
    mask = frontier_mask.convert('L')
    if original.size != mask.size:
        raise ValueError('Canvas and Frontier mask sizes differ')
    expected = get_dimensions(*original.size)
    if generated_padded.size != expected:
        raise ValueError(
            f'OpenAI returned {generated_padded.width}x{generated_padded.height}; '
            f'expected {expected[0]}x{expected[1]}. '
            'Refusing to resize, crop arbitrary content or publish.'
        )
    generated = generated_padded.convert('RGBA').crop((0, 0, *original.size))
    keep_original = mask.point(lambda p: 255 if p == 0 else 0)
    return Image.composite(original, generated, keep_original)


def generate(canvas_path: Path, mask_path: Path, prompt_path: Path, output_path: Path,
             client=None, model=None, quality=None):
    """Call the paid OpenAI edit endpoint and atomically write one native-size PNG."""
    model = model or os.getenv('OPENAI_IMAGE_MODEL', 'gpt-image-2.5-sunburst')
    quality = quality or os.getenv('OPENAI_IMAGE_QUALITY', 'high')
    if model not in MODEL_ALLOWLIST:
        raise ValueError(
            f'{model!r} is not supported for native custom-size edits here. '
            'Use gpt-image-2.5-sunburst or gpt-image-2.5-flare.'
        )
    if quality not in QUALITY_ALLOWLIST:
        raise ValueError(f'Invalid OPENAI_IMAGE_QUALITY={quality!r}')
    if client is None and not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is missing; configure it in an untracked .env file')
    prompt = prompt_path.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('Prompt file is empty')

    with Image.open(canvas_path) as original_file, Image.open(mask_path) as mask_file:
        original = original_file.convert('RGBA')
        mask = mask_file.convert('L')
    padded_canvas, api_mask = create_padded_inputs(original, mask)
    if client is None:
        from openai import OpenAI  # import only for live API runs
        client = OpenAI(timeout=240.0, max_retries=2)

    canvas_bytes, mask_bytes = BytesIO(), BytesIO()
    padded_canvas.save(canvas_bytes, format='PNG')
    api_mask.save(mask_bytes, format='PNG')
    canvas_bytes.seek(0)
    mask_bytes.seek(0)
    canvas_bytes.name = 'canvas.png'
    mask_bytes.name = 'mask.png'
    print(f'OpenAI Image Edit: {model}, {quality}, {padded_canvas.width}x{padded_canvas.height} '
          '(no resampling)', flush=True)

    response = client.images.edit(
        model=model,
        image=canvas_bytes,
        mask=mask_bytes,
        prompt=(prompt + '\n\nKeep the existing terrain, camera perspective, '
                'lighting, style and visible geography perfectly continuous '
                'with the fixed boundary. Edit only the transparent masked region. '
                'No labels, borders, frames, or UI.'),
        size=f'{padded_canvas.width}x{padded_canvas.height}',
        quality=quality,
        output_format='png',
    )
    if not response.data or not response.data[0].b64_json:
        raise RuntimeError('OpenAI edit returned no image data')
    image_bytes = base64.b64decode(response.data[0].b64_json, validate=True)
    with Image.open(BytesIO(image_bytes)) as model_result:
        final = restore_original_fixed_pixels(original, model_result, mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='.generated-', suffix='.png', dir=output_path.parent)
    os.close(fd)
    try:
        final.convert('RGB').save(tmp_name, format='PNG')
        os.replace(tmp_name, output_path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)

    # No prompts, images, or API keys copied to log; prompt lives under input/.
    log = {
        'model': model, 'quality': quality,
        'canvas': [original.width, original.height],
        'requested': [padded_canvas.width, padded_canvas.height],
        'output': str(output_path),
        'resampling': False,
        'fixed_pixel_restore': True,
    }
    output_path.parent.joinpath('provider_log.json').write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return output_path


def main():
    generate(
        Path(os.environ['FRONTIER_CANVAS']),
        Path(os.environ['FRONTIER_MASK']),
        Path(os.environ['FRONTIER_PROMPT']),
        Path(os.environ['FRONTIER_OUTPUT']),
    )


if __name__ == '__main__':
    main()
