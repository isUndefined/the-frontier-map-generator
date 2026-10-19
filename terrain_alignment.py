"""Register generated terrain against the fixed source before looking for a seam.

An affine transform corrects translation, rotation and different X/Y scales.
Only a well-supported transform is accepted; this cannot recreate invented terrain.
"""
from __future__ import annotations
import numpy as np


class AlignmentError(ValueError):
    pass



def _local_seam_warp(cv2, aligned, reference, fixed, side, core, overlap):
    """Refine terrain geometry locally near the seam using dense optical flow.

    The global affine stage handles translation/rotation/scale. This stage only
    applies a small, smoothly decaying non-rigid correction near the shared
    overlap so roads, rivers and ridge lines can meet more naturally.

    Returns (warped, report). If refinement is unsafe or unhelpful, returns the
    original aligned image with an explanatory report.
    """
    h, w = fixed.shape
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    gray_src = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)

    # Derive the real fixed-region boundary from the mask itself instead of
    # assuming a seam coordinate from canvas geometry. This is important for
    # east/west jobs where generation_canvas can be wider than one tile because
    # of the extra context area.
    ys_fixed, xs_fixed = np.nonzero(fixed)
    if not len(xs_fixed):
        return aligned, {
            'applied': False,
            'reason': 'no_fixed_pixels_for_seam',
        }

    band = max(48, min(int(overlap * 1.25), max(w, h) // 4))
    roi = np.zeros((h, w), np.uint8)

    if side == 'east':
        seam = int(xs_fixed.min())
        x0, x1 = max(0, seam - band), min(w, seam + overlap)
        roi[:, x0:x1] = 255
    elif side == 'west':
        seam = int(xs_fixed.max())
        x0, x1 = max(0, seam - overlap + 1), min(w, seam + band + 1)
        roi[:, x0:x1] = 255
    elif side == 'north':
        seam = int(ys_fixed.max())
        y0, y1 = max(0, seam - overlap + 1), min(h, seam + band + 1)
        roi[y0:y1, :] = 255
    else:  # south
        seam = int(ys_fixed.min())
        y0, y1 = max(0, seam - band), min(h, seam + overlap)
        roi[y0:y1, :] = 255

    # Dense optical flow is computed on a mildly blurred image to prefer
    # large terrain structures over individual tree/roof pixels.
    ref_blur = cv2.GaussianBlur(gray_ref, (0, 0), 1.4)
    src_blur = cv2.GaussianBlur(gray_src, (0, 0), 1.4)

    flow = cv2.calcOpticalFlowFarneback(
        ref_blur, src_blur, None,
        pyr_scale=0.5, levels=4, winsize=31, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=0
    ).astype(np.float32)

    # Flow maps reference coordinates toward the current source. Reject large
    # deformations: local refinement is meant to be subtle.
    mag = np.linalg.norm(flow, axis=2)
    valid = (roi > 0) & fixed
    if int(valid.sum()) < 500:
        return aligned, {
            'applied': False,
            'reason': 'too_few_fixed_pixels_near_seam',
        }

    p95 = float(np.percentile(mag[valid], 95))
    max_allowed = max(3.0, min(18.0, overlap * 0.08))
    if p95 > max_allowed:
        return aligned, {
            'applied': False,
            'reason': 'local_flow_too_large',
            'flow_p95_px': p95,
            'max_allowed_px': max_allowed,
        }

    # Smooth the float32 displacement field. Some OpenCV 5 builds restrict
    # medianBlur() to CV_8U, while optical flow is CV_32F. GaussianBlur()
    # supports float32 directly and avoids quantizing the displacement vectors.
    fx = cv2.GaussianBlur(flow[..., 0], (9, 9), 0)
    fy = cv2.GaussianBlur(flow[..., 1], (9, 9), 0)

    yy, xx = np.indices((h, w), dtype=np.float32)

    # Weight = 1 close to the fixed overlap / seam, then smoothly decay into
    # generated territory. Correction outside the seam neighborhood is zero.
    weight = np.zeros((h, w), np.float32)
    transition = max(64, min(int(overlap * 1.5), 220))

    if side == 'east':
        # Existing/fixed terrain starts at `seam` and extends right; generated
        # territory is left of it, so fade correction as we move left.
        seam_x = float(seam)
        dist = np.maximum(0.0, seam_x - xx)
        weight = np.clip(1.0 - dist / transition, 0.0, 1.0)
    elif side == 'west':
        # Existing/fixed terrain ends at `seam`; generated territory is right.
        seam_x = float(seam)
        dist = np.maximum(0.0, xx - seam_x)
        weight = np.clip(1.0 - dist / transition, 0.0, 1.0)
    elif side == 'north':
        # Existing/fixed terrain ends at `seam`; generated territory is below.
        seam_y = float(seam)
        dist = np.maximum(0.0, yy - seam_y)
        weight = np.clip(1.0 - dist / transition, 0.0, 1.0)
    else:  # south
        # Existing/fixed terrain starts at `seam`; generated territory is above.
        seam_y = float(seam)
        dist = np.maximum(0.0, seam_y - yy)
        weight = np.clip(1.0 - dist / transition, 0.0, 1.0)

    # Smoothstep and restrict to the ROI expanded slightly into new terrain.
    weight = weight * weight * (3.0 - 2.0 * weight)
    weight *= (roi > 0).astype(np.float32)

    # Build a confidence map from local structure. Optical flow is useful only
    # where the image contains stable gradients/corners; uniform fields, mist,
    # water glare and repetitive forest texture should receive less deformation.
    gx = cv2.Sobel(ref_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # Normalize robustly so a few very strong edges do not dominate.
    g95 = float(np.percentile(grad[roi > 0], 95)) if np.any(roi > 0) else 1.0
    if g95 <= 1e-6:
        g95 = 1.0
    structure = np.clip(grad / g95, 0.0, 1.0)

    # Corner response adds confidence for junctions and terrain landmarks.
    corner = cv2.cornerMinEigenVal(ref_blur, blockSize=5, ksize=3)
    c95 = float(np.percentile(corner[roi > 0], 95)) if np.any(roi > 0) else 1.0
    if c95 <= 1e-9:
        c95 = 1.0
    corners = np.clip(corner / c95, 0.0, 1.0)

    # Combine broad gradient evidence with stronger landmark evidence.
    confidence = np.clip(0.55 * structure + 0.45 * corners, 0.0, 1.0)
    confidence = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 4.0)

    # Keep a small floor so large terrain masses can still receive gentle
    # correction, but suppress aggressive warping in low-confidence regions.
    confidence = 0.15 + 0.85 * confidence

    # Never apply more than the conservative cap.
    scale = np.minimum(1.0, max_allowed / np.maximum(mag, 1e-6))
    adaptive_weight = weight * confidence
    fx = fx * scale * adaptive_weight
    fy = fy * scale * adaptive_weight

    map_x = xx + fx
    map_y = yy + fy

    warped = cv2.remap(
        aligned, map_x.astype(np.float32), map_y.astype(np.float32),
        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
    )

    before = float(np.abs(aligned.astype(np.float32) - reference.astype(np.float32))[valid].mean())
    after = float(np.abs(warped.astype(np.float32) - reference.astype(np.float32))[valid].mean())

    # Require measurable improvement; otherwise preserve affine result.
    if after >= before * 0.985:
        return aligned, {
            'applied': False,
            'reason': 'no_meaningful_improvement',
            'mae_before': before,
            'mae_after': after,
            'flow_p95_px': p95,
        }

    return warped, {
        'applied': True,
        'method': 'farneback_adaptive_local_seam',
        'transition_px': transition,
        'seam_coordinate_px': int(seam),
        'flow_p95_px': p95,
        'confidence_mean': float(confidence[roi > 0].mean()),
        'confidence_p10': float(np.percentile(confidence[roi > 0], 10)),
        'confidence_p90': float(np.percentile(confidence[roi > 0], 90)),
        'mae_before': before,
        'mae_after': after,
    }



def _translation_fallback(cv2, generated, reference, fixed, initial_mae, overlap):
    """Bounded translation-only fallback using the fixed overlap itself.

    Unlike global phase correlation, this searches only a small ±N pixel window
    around the fixed overlap, preventing false locks onto similar terrain far
    away (for example repetitive forest or river texture).
    """
    h, w = fixed.shape
    ys, xs = np.nonzero(fixed)
    if not len(xs):
        raise AlignmentError('No fixed terrain available for translation fallback')

    max_shift = int(round(max(4.0, min(20.0, float(overlap) * 0.08))))

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gen_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Work on terrain structure rather than absolute brightness.
    def structure(gray):
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    ref_s = structure(ref_gray)
    gen_s = structure(gen_gray)

    # Tight bounding box of the fixed area. MinFilter in the caller may shrink
    # it slightly; that is fine and actually avoids unreliable mask edges.
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    template = ref_s[y0:y1, x0:x1]
    if template.shape[0] < 24 or template.shape[1] < 24:
        raise AlignmentError('Fixed overlap is too small for bounded translation matching')
    if float(np.std(template)) < 1.0:
        raise AlignmentError('Fixed overlap has too little terrain detail for translation matching')

    # Pad generated structure so negative shifts are searchable too.
    padded = cv2.copyMakeBorder(
        gen_s, max_shift, max_shift, max_shift, max_shift,
        cv2.BORDER_REFLECT_101
    )

    # Search exactly ±max_shift around the expected overlap location.
    sx0 = x0
    sy0 = y0
    sx1 = x1 + 2 * max_shift
    sy1 = y1 + 2 * max_shift
    search = padded[sy0:sy1, sx0:sx1]

    expected_h = template.shape[0] + 2 * max_shift
    expected_w = template.shape[1] + 2 * max_shift
    if search.shape[0] != expected_h or search.shape[1] != expected_w:
        raise AlignmentError(
            f'Unexpected bounded-search geometry {search.shape[1]}x{search.shape[0]}'
        )

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, best = cv2.minMaxLoc(result)

    dx = float(best[0] - max_shift)
    dy = float(best[1] - max_shift)

    # Require a useful structural correlation. This threshold is intentionally
    # conservative; poor matches should remain drafts rather than be published.
    if score < 0.18:
        raise AlignmentError(
            f'Bounded translation correlation too weak ({score:.3f})'
        )

    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = xx + dx
    map_y = yy + dy

    shifted = cv2.remap(
        generated,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    after = float(
        np.abs(shifted.astype(np.float32) - reference.astype(np.float32))[fixed].mean()
    )

    # Require a real improvement in the fixed overlap.
    if after >= initial_mae * 0.97:
        raise AlignmentError(
            f'Bounded translation did not improve agreement enough '
            f'({initial_mae:.2f} -> {after:.2f})'
        )

    return shifted, {
        'method': 'bounded_template_translation',
        'dx_px': dx,
        'dy_px': dy,
        'correlation_score': float(score),
        'anchor_mae_before': initial_mae,
        'anchor_mae_after': after,
        'max_shift_px': max_shift,
        'search_window_px': [-max_shift, max_shift],
    }


def align_terrain(generated, reference, fixed, side, core, overlap, initial_mae):
    try:
        import cv2
    except ImportError as exc:
        raise AlignmentError(
            'Terrain alignment needs OpenCV. Run .venv/bin/python stitch_region_grid.py '
            'or install requirements.txt in your Python environment.') from exc
    cv2.setRNGSeed(0)
    mask = fixed.astype(np.uint8) * 255
    h, w = fixed.shape
    ys, xs = np.nonzero(fixed)
    if not len(xs):
        raise AlignmentError('No fixed terrain available for alignment')
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    gray_gen = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=15000, contrastThreshold=0.025)
    # A little outside the expected anchor allows displaced details to be found.
    margin = max(15, round(min(w, h) * .08))
    search = cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    kr, dr = sift.detectAndCompute(gray_ref, mask)
    kg, dg = sift.detectAndCompute(gray_gen, search)
    if dr is None or dg is None or len(dr) < 20 or len(dg) < 20:
        shifted, report = _translation_fallback(
            cv2, generated, reference, fixed, initial_mae, overlap
        )
        report['matches'] = 0
        report['status'] = 'accepted'
        return shifted, report
    pairs = cv2.BFMatcher().knnMatch(dr, dg, k=2)
    candidates = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
    # One generated detail cannot support several independent correspondences.
    unique = {}
    for match in sorted(candidates, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    matches = list(unique.values())
    if len(matches) < 20:
        shifted, report = _translation_fallback(
            cv2, generated, reference, fixed, initial_mae, overlap
        )
        report['matches'] = len(matches)
        report['status'] = 'accepted'
        return shifted, report
    a = np.float32([kr[m.queryIdx].pt for m in matches])
    b = np.float32([kg[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffine2D(a, b, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0, maxIters=10000,
                                         confidence=.999, refineIters=20)
    if matrix is None:
        raise AlignmentError('Terrain matches do not support a consistent transform')
    good = inliers.ravel().astype(bool)
    count = int(good.sum())
    if count < 20 or count / len(matches) < .3:
        raise AlignmentError(f'Inconsistent terrain: {count}/{len(matches)} matching details agree')
    points = a[good]
    span = np.ptp(points, axis=0)
    anchor_size = np.array([xs.max() - xs.min() + 1, ys.max() - ys.min() + 1])
    if np.any(span / anchor_size < .45):
        raise AlignmentError('Terrain matches are too concentrated to align the full anchor')
    # Require landmarks near the future seam, not just a distant copied town.
    if side == 'east':
        near = points[:, 0] < core + overlap
        along = 1
    elif side == 'west':
        near = points[:, 0] >= xs.max() - overlap
        along = 1
    elif side == 'north':
        near = points[:, 1] < overlap
        along = 0
    else:
        near = points[:, 1] >= core
        along = 0
    if near.sum() < 6 or np.ptp(points[near, along]) < .4 * anchor_size[along]:
        raise AlignmentError('Not enough matching terrain near the seam to preserve its geometry')
    scales = np.linalg.svd(matrix[:, :2], compute_uv=False)
    if np.linalg.det(matrix[:, :2]) <= 0 or scales.min() < .8 or scales.max() > 1.2:
        raise AlignmentError('Estimated terrain distortion is too large for a safe alignment')
    predicted = points @ matrix[:, :2].T + matrix[:, 2]
    residual = np.linalg.norm(predicted - b[good], axis=1)
    # The matrix maps desired/reference coordinates to sampling coordinates.
    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    map_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
    displacement = np.hypot(map_x - xx, map_y - yy)
    if float(np.percentile(displacement[fixed], 95)) > .15 * max(w, h):
        raise AlignmentError('Estimated terrain displacement is too large')
    # Fade the correction toward the distant boundary of the new territory.
    # The reference anchor is aligned fully; no cumulative map is transformed.
    if side == 'east':
        weight = np.clip(xx / max(1, core), 0, 1)
    elif side == 'west':
        weight = np.clip((w - 1 - xx) / max(1, core), 0, 1)
    elif side == 'north':
        weight = np.clip((h - 1 - yy) / max(1, core), 0, 1)
    else:
        weight = np.clip(yy / max(1, core), 0, 1)
    weight = weight * weight * (3 - 2 * weight)

    # Fade the affine correction near the borders perpendicular to the expansion
    # direction. Without this, a small rotation/shear can pull corners of the new
    # territory outside the source image even when the seam itself aligns well.
    edge_margin = max(32, min(int(overlap * 0.5), 128))
    if side in ('north', 'south'):
        edge_dist = np.minimum(xx, (w - 1) - xx)
    else:
        edge_dist = np.minimum(yy, (h - 1) - yy)

    edge_weight = np.clip(edge_dist / max(1.0, float(edge_margin)), 0.0, 1.0)
    edge_weight = edge_weight * edge_weight * (3.0 - 2.0 * edge_weight)

    # Preserve correction along the stitching direction all the way to the
    # corners. Only fade the perpendicular component that can push sampling
    # coordinates outside the source image.
    dx = map_x - xx
    dy = map_y - yy

    if side in ('north', 'south'):
        # Vertical stitching: keep dy everywhere so the shared horizontal seam
        # remains aligned even at the left/right corners. Fade only dx.
        dx *= edge_weight
    else:
        # Horizontal stitching: keep dx everywhere so the shared vertical seam
        # remains aligned even at the top/bottom corners. Fade only dy.
        dy *= edge_weight

    map_x = (xx + weight * dx).astype(np.float32)
    map_y = (yy + weight * dy).astype(np.float32)
    outside = np.maximum.reduce([-map_x, map_x - (w - 1), -map_y, map_y - (h - 1)])

    # Allow a small interpolation border relative to overlap size. OpenCV remap
    # uses reflected pixels there, so a few pixels are safe; large excursions
    # still indicate an unreliable transform.
    max_border = max(6.0, min(24.0, float(overlap) * 0.08))
    outside_new = float(outside[~fixed].max(initial=0))
    if outside_new > max_border:
        raise AlignmentError(
            f'Alignment would leave missing pixels in the new territory '
            f'({outside_new:.2f}px > allowed {max_border:.2f}px)'
        )
    aligned = cv2.remap(generated, map_x, map_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101)
    after = float(np.abs(aligned.astype(np.float32) - reference.astype(np.float32))[fixed].mean())
    if after >= initial_mae * .9:
        raise AlignmentError(f'Geometric alignment did not improve agreement enough ({initial_mae:.2f} -> {after:.2f})')

    # Second stage: a conservative local non-rigid warp near the seam.
    refined, local_report = _local_seam_warp(
        cv2, aligned, reference, fixed, side, core, overlap
    )
    final_after = float(
        np.abs(refined.astype(np.float32) - reference.astype(np.float32))[fixed].mean()
    )

    return refined, {
        'method': 'sift_affine_adaptivewarp' if local_report.get('applied') else 'sift_affine',
        'matches': len(matches), 'inliers': count,
        'seam_inliers': int(near.sum()), 'reference_to_generated': matrix.tolist(),
        'median_landmark_error_px': float(np.median(residual)),
        'p95_landmark_error_px': float(np.percentile(residual, 95)),
        'anchor_mae_before': initial_mae, 'anchor_mae_after': final_after,
        'affine_anchor_mae_after': after,
        'local_warp': local_report,
        'max_reflected_border_px': float(max(0, outside[~fixed].max(initial=0))),
        'allowed_reflected_border_px': max_border,
        'affine_edge_fade_px': edge_margin,
        'affine_edge_fade_mode': 'perpendicular_component_only',
        'note': 'Local warp is conservative and only accepted when it improves seam agreement.',
    }
