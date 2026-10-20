"""Bounded tonal adaptation at a generated tile's editable seam.

Only generated pixels are adjusted. Existing world pixels are never recolored,
resized, or blended with geometrically different objects.
"""
import numpy as np


def tone_match_generated(generated, locked, x0, x1, y0, y1, radius=96):
    """Correct modest exposure/color mismatch in last radius px of the core.

    Inputs are oriented with the neighbor on the right. Compare colors at
    matching world coordinates inside editable overlap; cap RGB gain/offset.
    """
    if generated.shape != locked.shape or generated.ndim != 3:
        raise ValueError('Tone-match input shapes differ')
    sample = generated[y0:y1, x0:x1].astype(np.float32)
    reference = locked[y0:y1, x0:x1].astype(np.float32)
    if sample.size == 0:
        return generated.copy(), {'enabled': False}
    src_med = np.median(sample, axis=(0, 1))
    ref_med = np.median(reference, axis=(0, 1))
    src_std = np.std(sample, axis=(0, 1))
    ref_std = np.std(reference, axis=(0, 1))
    gains = np.clip(ref_std / np.maximum(src_std, 3.0), .94, 1.06)
    offsets = np.clip(ref_med - src_med * gains, -18.0, 18.0)

    start = max(0, x0 - max(0, int(radius)))
    xx = np.arange(start, x1, dtype=np.float32)
    progress = np.clip((xx - start) / max(1, x0 - start), 0, 1)
    weights = progress * progress * (3 - 2 * progress)
    result = generated.astype(np.float32)
    adjustment = np.clip(result[:, start:x1] * gains[None, None, :] +
                         offsets[None, None, :], 0, 255)
    result[:, start:x1] += weights[None, :, None] * (adjustment - result[:, start:x1])
    return np.uint8(np.clip(np.rint(result), 0, 255)), {
        'enabled': True,
        'radius_px': int(x0 - start),
        'gain_rgb': [round(float(v), 4) for v in gains],
        'offset_rgb': [round(float(v), 2) for v in offsets],
    }
