#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diff_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).sum(axis=2)


def min_vertical_seam(cost: np.ndarray) -> np.ndarray:
    h, w = cost.shape
    dp = cost.copy()
    parent = np.zeros((h, w), dtype=np.int32)

    big = np.float32(1e20)
    xs = np.arange(w)
    for y in range(1, h):
        prev = dp[y - 1]
        left = np.empty_like(prev)
        right = np.empty_like(prev)
        left[0] = big
        left[1:] = prev[:-1]
        right[-1] = big
        right[:-1] = prev[1:]

        stacked = np.stack([left, prev, right], axis=0)
        idx = np.argmin(stacked, axis=0)
        parent[y] = np.clip(xs + (idx - 1), 0, w - 1)
        dp[y] = cost[y] + stacked[idx, xs]

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for y in range(h - 1, 0, -1):
        seam[y - 1] = parent[y, seam[y]]
    return seam


def merge_overlap_left_right(left_img: np.ndarray, right_img: np.ndarray, feather: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    if left_img.shape != right_img.shape:
        raise ValueError(f"Overlap images must have same shape, got {left_img.shape} vs {right_img.shape}")

    h, w, _ = left_img.shape
    cost = diff_cost(left_img, right_img)
    seam = min_vertical_seam(cost)
    merged = np.empty_like(left_img)
    feather = max(0, int(feather))

    for y in range(h):
        x = int(seam[y])
        if feather == 0:
            merged[y, :x] = left_img[y, :x]
            merged[y, x:] = right_img[y, x:]
            continue

        x0 = max(0, x - feather)
        x1 = min(w, x + feather + 1)
        if x0 > 0:
            merged[y, :x0] = left_img[y, :x0]
        if x1 < w:
            merged[y, x1:] = right_img[y, x1:]

        blend_w = x1 - x0
        if blend_w > 0:
            alpha = np.linspace(0.0, 1.0, blend_w, dtype=np.float32)[:, None]
            l = left_img[y, x0:x1].astype(np.float32)
            r = right_img[y, x0:x1].astype(np.float32)
            merged[y, x0:x1] = np.clip(l * (1.0 - alpha) + r * alpha, 0, 255).astype(np.uint8)

    return merged, seam


def seam_overlay_preview(left_img: np.ndarray, right_img: np.ndarray, seam: np.ndarray) -> Image.Image:
    preview = ((left_img.astype(np.float32) * 0.5) + (right_img.astype(np.float32) * 0.5)).astype(np.uint8)
    h, w, _ = preview.shape
    for y in range(h):
        x = int(seam[y])
        preview[y, max(0, x - 1): min(w, x + 2)] = np.array([255, 70, 70], dtype=np.uint8)
    return Image.fromarray(preview)


def scaled_layout(job: dict, gen_w: int, gen_h: int) -> dict:
    expected_w = int(job["total_generation_width"])
    expected_h = int(job["native_height"])
    new_core = int(job["new_core_width"])
    overlap = int(job["overlap_width"])
    context = int(job["context_width"])

    sx = gen_w / expected_w
    sy = gen_h / expected_h

    # The generator often returns a proportionally smaller canvas.
    # Scale X-layout coordinates to the actual generated width; never resize generated.png.
    new_core_actual = max(1, round(new_core * sx))
    overlap_actual = max(8, round(overlap * sx))
    context_actual = gen_w - new_core_actual - overlap_actual

    if context_actual < 8:
        # Fallback based on proportional context if cumulative rounding was bad.
        context_actual = max(8, round(context * sx))
        new_core_actual = gen_w - overlap_actual - context_actual

    aspect_drift = abs(sx - sy) / max(sx, sy)
    if aspect_drift > 0.02:
        raise RuntimeError(
            f"Generated image changed aspect ratio too much: expected {expected_w}x{expected_h}, "
            f"got {gen_w}x{gen_h}; sx={sx:.5f}, sy={sy:.5f}. "
            "This is not a simple proportional native-size change. Regenerate the image."
        )

    return {
        "sx": sx,
        "sy": sy,
        "expected_width": expected_w,
        "expected_height": expected_h,
        "new_core_width": new_core_actual,
        "overlap_width": overlap_actual,
        "context_width": context_actual,
    }


def stitch(job_dir: Path, generated_path: Path, feather: int = 4) -> None:
    job = load_json(job_dir / "job.json")
    direction = job["direction"]
    source_native_path = Path(job["source_region_native"])

    gen_img = Image.open(generated_path).convert("RGB")
    gen_w, gen_h = gen_img.size
    gen = np.array(gen_img)

    layout = scaled_layout(job, gen_w, gen_h)
    new_core_width = layout["new_core_width"]
    overlap_width = layout["overlap_width"]
    context_width = layout["context_width"]

    # Resize ONLY the existing source map to the actual generated height.
    # generated.png itself is never resized.
    src_img = Image.open(source_native_path).convert("RGB")
    src_w_actual = max(1, round(src_img.width * (gen_h / src_img.height)))
    src_img = src_img.resize((src_w_actual, gen_h), Image.LANCZOS)
    src = np.array(src_img)

    if src.shape[1] <= overlap_width:
        raise RuntimeError(
            f"Source region is too narrow after native-height scaling: source={src.shape[1]}px, overlap={overlap_width}px"
        )

    out_dir = job_dir / "stitched"
    out_dir.mkdir(parents=True, exist_ok=True)

    if direction == "left":
        # actual generator layout: [NEW_CORE][GEN_OVERLAP][SOURCE_CONTEXT]
        o0 = new_core_width
        o1 = o0 + overlap_width
        if o1 > gen_w:
            raise RuntimeError(f"Calculated overlap exceeds generated width: {o1} > {gen_w}")

        gen_core = gen[:, :o0]
        gen_overlap = gen[:, o0:o1]
        src_overlap = src[:, :overlap_width]
        src_tail = src[:, overlap_width:]
        merged_overlap, seam = merge_overlap_left_right(gen_overlap, src_overlap, feather=feather)
        final = np.concatenate([gen_core, merged_overlap, src_tail], axis=1)
        nominal_x = new_core_width
    else:
        # actual generator layout: [SOURCE_CONTEXT][GEN_OVERLAP][NEW_CORE]
        o0 = context_width
        o1 = o0 + overlap_width
        if o1 > gen_w:
            raise RuntimeError(f"Calculated overlap exceeds generated width: {o1} > {gen_w}")

        gen_overlap = gen[:, o0:o1]
        gen_core = gen[:, o1:]
        src_head = src[:, : src.shape[1] - overlap_width]
        src_overlap = src[:, src.shape[1] - overlap_width :]
        merged_overlap, seam = merge_overlap_left_right(src_overlap, gen_overlap, feather=feather)
        final = np.concatenate([src_head, merged_overlap, gen_core], axis=1)
        nominal_x = src_head.shape[1]

    final_path = out_dir / "final_native.png"
    Image.fromarray(final).save(final_path)

    seam_preview = seam_overlay_preview(gen_overlap, src_overlap, seam)
    seam_preview.save(out_dir / "seam_preview.png")

    final_preview = final.copy()
    final_preview[:, max(0, nominal_x - 1): min(final_preview.shape[1], nominal_x + 1)] = np.array([255, 210, 60], dtype=np.uint8)
    Image.fromarray(final_preview).save(out_dir / "final_native_preview.png")

    meta = {
        "generated_size": [gen_w, gen_h],
        "expected_generation_size": [layout["expected_width"], layout["expected_height"]],
        "scale_x": layout["sx"],
        "scale_y": layout["sy"],
        "actual_new_core_width": new_core_width,
        "actual_overlap_width": overlap_width,
        "actual_context_width": context_width,
        "source_scaled_size": [src.shape[1], src.shape[0]],
        "feather": feather,
        "final_size": [final.shape[1], final.shape[0]],
    }
    with open(out_dir / "stitch_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Generated actual size          : {gen_w}x{gen_h}")
    print(f"Expected generation size      : {layout['expected_width']}x{layout['expected_height']}")
    print(f"Scale                          : x={layout['sx']:.6f}, y={layout['sy']:.6f}")
    print(f"Actual NEW_CORE / OVERLAP / CTX: {new_core_width} / {overlap_width} / {context_width}")
    print(f"Stitched SHARED OVERLAP result: {final_path}")
    print(f"Seam preview                  : {out_dir / 'seam_preview.png'}")
    print(f"Final preview                 : {out_dir / 'final_native_preview.png'}")
    print(f"Final size                    : {final.shape[1]}x{final.shape[0]}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stitch SHARED OVERLAP output without resizing the generated artwork.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stitch")
    s.add_argument("--job", required=True)
    s.add_argument("--generated", required=True)
    s.add_argument("--feather", type=int, default=4)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "stitch":
        stitch(Path(args.job), Path(args.generated), feather=args.feather)
