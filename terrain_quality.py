"""Conservative copied-landform detection for independent Frontier art tiles.

Features are measured at native pixel resolution. Areas legitimately shared
with neighboring tiles are excluded from the candidate image before matching,
so a correct overlap cannot trigger the duplication gate.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tile_grid import discover, geometry, load_json


def _rgb(path, size):
    with Image.open(path) as im:
        if im.size != size:
            raise ValueError(f'{path}: size {im.size} differs from {size}')
        return np.asarray(im.convert('RGB'))


def detect_copied_terrain(jobs, target, tile_path):
    """Find strong, spatially wide copied terrain beyond physical overlap bands.

    A clear result is not proof of beautiful art; it only means this particular
    duplicate-pattern test did not find enough reliable evidence to reject it.
    """
    jobs = Path(jobs).resolve()
    layout = geometry(jobs)
    width, height, ov = (int(layout[k]) for k in ('width', 'height', 'overlap'))
    tx, ty = int(target['x']), int(target['y'])
    refs = [(pos, r) for pos, r in discover(jobs).items()
            if pos != (tx, ty) and max(abs(pos[0] - tx), abs(pos[1] - ty)) <= 1]
    if not refs:
        return []

    # Ignore only actual shared overlap strips, including diagonal corners.
    x0 = ov if any(x < tx for (x, y), _ in refs) else 0
    x1 = width - ov if any(x > tx for (x, y), _ in refs) else width
    y0 = ov if any(y < ty for (x, y), _ in refs) else 0
    y1 = height - ov if any(y > ty for (x, y), _ in refs) else height
    if x1 - x0 < 200 or y1 - y0 < 140:
        return []  # insufficient untouched territory for a wide-area test

    image = _rgb(tile_path, (width, height))[y0:y1, x0:x1]
    sift = cv2.SIFT_create(nfeatures=5500, contrastThreshold=0.028)
    features, descriptors = sift.detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None
    )
    if descriptors is None or len(features) < 35:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = []
    for pos, ref in refs:
        reference = _rgb(ref['image'], (width, height))
        keys, ref_descriptors = sift.detectAndCompute(
            cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None
        )
        if ref_descriptors is None or len(keys) < 35:
            continue
        pairs = bf.knnMatch(descriptors, ref_descriptors, k=2)
        good = [m for m, n in pairs if m.distance < .75 * n.distance]
        if len(good) < 35:
            continue

        src = np.float32([features[m.queryIdx].pt for m in good])
        dst = np.float32([keys[m.trainIdx].pt for m in good])
        cv2.setRNGSeed(0)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
            maxIters=3500, confidence=.999,
        )
        if matrix is None or inliers is None:
            continue
        accepted = inliers.reshape(-1).astype(bool)
        count = int(accepted.sum())
        if count < 35 or count / len(good) < .45:
            continue

        a, b = float(matrix[0, 0]), float(matrix[1, 0])
        scale = (a*a + b*b) ** .5
        if not (0.985 <= scale <= 1.015 and abs(b) <= .025):
            continue
        span_x, span_y = np.ptp(src[accepted], axis=0)
        core_w, core_h = x1 - x0, y1 - y0
        if (span_x < max(160, .20 * core_w)
                or span_y < max(120, .15 * core_h)
                or span_x * span_y < .10 * core_w * core_h):
            continue

        matches.append({
            'source_tile': ref['id'],
            'source_grid': list(pos),
            'inliers': count, 'feature_matches': len(good),
            'inlier_ratio': round(count / len(good), 3),
            'translation_px': [
                round(float(matrix[0, 2] - x0 * matrix[0, 0] + y0 * matrix[0, 1]), 1),
                round(float(matrix[1, 2] - x0 * matrix[1, 0] - y0 * matrix[1, 1]), 1),
            ],
            'covered_area_px': [round(float(span_x)), round(float(span_y))],
            'scale': round(scale, 4),
            'reason': 'Repeated wide-area image features outside the legitimate overlap',
        })
    return matches


def audit_tile(jobs, job, tile_path, stitch_metadata=None, allow_weak_alignment=False):
    """Compute a machine-readable QA report for new or existing published art."""
    jobs = Path(jobs).resolve()
    folder = jobs / job
    metadata = folder / 'grid_job.json'
    if not metadata.is_file():
        raise ValueError(f'No prepared job: {job}')
    target = load_json(metadata)['target']
    if stitch_metadata is None:
        path = folder / 'stitch_meta.json'
        stitch_metadata = load_json(path) if path.is_file() else {}
    alignment = stitch_metadata.get('alignment') or {}
    method = alignment.get('method') or stitch_metadata.get('alignment_method') or 'unknown'
    unreliable = (alignment.get('status') != 'accepted'
                  or method in ('unaligned_chatgpt_fallback', 'unresolved', 'unknown'))

    duplicates = detect_copied_terrain(jobs, target, tile_path)
    issues = []
    if duplicates:
        issues.append('duplicated_terrain')
    if unreliable and not allow_weak_alignment:
        issues.append('weak_alignment')

    return {
        'format': 'frontier_tile_qa_v1',
        'job': job,
        'grid': target,
        'status': 'rejected' if issues else 'passed',
        'issues': issues,
        'duplicate_matches': duplicates,
        'alignment': {
            'method': method,
            'status': alignment.get('status', 'unknown'),
            'reason': alignment.get('reason'),
        },
        'weak_alignment_override': bool(unreliable and allow_weak_alignment),
        'tile': str(tile_path),
    }
