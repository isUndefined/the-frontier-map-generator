#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


DEFAULT_NATIVE_HEIGHT = 853
DEFAULT_OVERLAP = 256
DEFAULT_CONTEXT = 700


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resize_to_height(im: Image.Image, target_h: int) -> Image.Image:
    w, h = im.size
    if h == target_h:
        return im.copy()
    new_w = max(1, round(w * target_h / h))
    return im.resize((new_w, target_h), Image.LANCZOS)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_prepare_overview(canvas: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    h, w, _ = canvas.shape
    vis = canvas.copy()

    # tint generated zone
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    tint = np.zeros_like(vis)
    tint[..., 0] = 160
    tint[..., 1] = 110
    tint[..., 2] = 30
    alpha = (mask3.astype(np.float32) / 255.0) * 0.24
    vis = (vis.astype(np.float32) * (1 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)

    # draw delimiter lines
    line_color = np.array([230, 200, 120], dtype=np.uint8)
    white = np.array([235, 235, 235], dtype=np.uint8)
    for x in range(0, w, max(1, w // 20)):
        vis[0:3, x:x+1] = white
    vis[0:3, :] = white
    vis[-3:, :] = white

    Image.fromarray(vis).save(out_path)



def build_prompt(direction: str, new_core_width: int, overlap_width: int, context_width: int, native_height: int, total_width: int) -> str:
    if direction == "left":
        layout = "[ NEW CORE ][ SHARED OVERLAP ][ SOURCE CONTEXT ]"
        extend = "left"
        source_side = "right"
    else:
        layout = "[ SOURCE CONTEXT ][ SHARED OVERLAP ][ NEW CORE ]"
        extend = "right"
        source_side = "left"

    return f"""Create one continuous regional map expansion for The Frontier.

CANVAS:
- Size: {total_width}x{native_height}
- Layout: {layout}
- NEW CORE: {new_core_width}px
- SHARED OVERLAP: {overlap_width}px
- SOURCE CONTEXT: {context_width}px
- Expansion direction: {extend}
- Existing Ravenfall context is on the {source_side} side.

GOAL:
Extend the existing Ravenfall Valley landscape naturally toward the {extend}.
The full image must read as one uninterrupted regional map.

SHARED OVERLAP:
The SHARED OVERLAP is intentionally generated together with the NEW CORE.
It must visually match the neighboring SOURCE CONTEXT so the two versions can later be stitched inside this overlap area.
Do not create a visible vertical boundary at either side of the overlap.

CONTINUITY:
- same high oblique 3/4 aerial regional map perspective
- same painterly realistic medieval frontier style
- same terrain scale
- same forest density logic
- same road scale
- same water scale
- same lighting direction
- same atmospheric depth
- continue ridges, forests, roads, rivers, cliffs and valleys naturally
- maintain consistent object scale and perspective

NEW REGION CHARACTER:
- northern temperate frontier landscape
- conifer forests dominant
- open forest patches and natural meadows
- rolling terrain with rocky ridges and stony outcrops
- sparse frontier traces
- no large city
- at most one subtle remote landmark

STRICT:
- no UI
- no labels
- no borders
- no map markers
- no text
- no mirrored copy of Ravenfall
- no hard seam
- no abrupt color or density change
- do not treat SOURCE CONTEXT as a separate panel

FINAL RESULT:
A seamless neighboring region where NEW CORE flows through SHARED OVERLAP into SOURCE CONTEXT as one continuous handcrafted world map scene.
"""


def prepare_job(
    source_path: Path,
    job_dir: Path,
    direction: str,
    add_width_native: int,
    native_height: int,
    overlap_width: int,
    context_width: int,
) -> None:
    if direction not in {"left", "right"}:
        raise ValueError("direction must be left or right")
    if add_width_native <= 0:
        raise ValueError("add_width_native must be > 0")
    if overlap_width <= 0:
        raise ValueError("overlap_width must be > 0")
    if context_width <= overlap_width:
        raise ValueError("context_width must be greater than overlap_width")

    ensure_dir(job_dir)
    input_dir = job_dir / "input"
    preview_dir = job_dir / "preview"
    ensure_dir(input_dir)
    ensure_dir(preview_dir)

    src_orig = Image.open(source_path).convert("RGB")
    source_orig_size = src_orig.size
    src = resize_to_height(src_orig, native_height)
    src_np = np.array(src)
    h, src_w, _ = src_np.shape

    if context_width > src_w:
        context_width = src_w
    if overlap_width >= context_width:
        raise ValueError("After clamping, overlap_width must still be smaller than context_width")

    total_w = add_width_native + overlap_width + context_width
    canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
    mask = np.zeros((h, total_w), dtype=np.uint8)

    if direction == "left":
        # Final layout after stitch: [NEW_CORE][MERGED_OVERLAP][SOURCE_TAIL]
        # Generator layout:          [NEW_CORE][SHARED_OVERLAP][SOURCE_CONTEXT]
        source_context = src_np[:, :context_width]
        canvas[:, add_width_native + overlap_width : total_w] = source_context
        mask[:, : add_width_native + overlap_width] = 255
        source_anchor_side = "left_edge"
    else:
        # Final layout after stitch: [SOURCE_HEAD][MERGED_OVERLAP][NEW_CORE]
        # Generator layout:          [SOURCE_CONTEXT][SHARED_OVERLAP][NEW_CORE]
        source_context = src_np[:, src_w - context_width :]
        canvas[:, :context_width] = source_context
        mask[:, context_width:] = 255
        source_anchor_side = "right_edge"

    generation_canvas_path = input_dir / "generation_canvas.png"
    generation_mask_path = input_dir / "generation_mask.png"
    source_native_path = input_dir / "source_region_native.png"
    source_original_copy_path = input_dir / "source_region.png"
    prompt_path = input_dir / "prompt.txt"

    Image.fromarray(canvas).save(generation_canvas_path)
    Image.fromarray(mask).save(generation_mask_path)
    src.save(source_native_path)
    src_orig.save(source_original_copy_path)

    prompt = build_prompt(
        direction=direction,
        new_core_width=add_width_native,
        overlap_width=overlap_width,
        context_width=context_width,
        native_height=h,
        total_width=total_w,
    )
    prompt_path.write_text(prompt, encoding="utf-8")

    make_prepare_overview(canvas, mask, preview_dir / "prepare_overview.png")

    boundary_reference = None
    if direction == "left":
        boundary_reference = src_np[:, : max(overlap_width, min(context_width, 320))]
    else:
        boundary_reference = src_np[:, src_w - max(overlap_width, min(context_width, 320)) :]
    Image.fromarray(boundary_reference).save(preview_dir / "boundary_reference.png")

    job = {
        "version": 3,
        "method": "shared_overlap",
        "direction": direction,
        "source_region": str(source_original_copy_path),
        "source_region_native": str(source_native_path),
        "generation_canvas": str(generation_canvas_path),
        "generation_mask": str(generation_mask_path),
        "prompt": str(prompt_path),
        "source_original_size": {"width": source_orig_size[0], "height": source_orig_size[1]},
        "source_native_size": {"width": src_w, "height": h},
        "native_height": h,
        "new_core_width": add_width_native,
        "overlap_width": overlap_width,
        "context_width": context_width,
        "total_generation_width": total_w,
        "source_anchor_side": source_anchor_side,
        "notes": {
            "left": "For left expansion, the generator sees left-edge context from the existing region on the right side of the canvas.",
            "right": "For right expansion, the generator sees right-edge context from the existing region on the left side of the canvas.",
        },
    }
    save_json(job_dir / "job.json", job)

    print(f"Prepared SHARED OVERLAP job: {job_dir}")
    print(f" direction       : {direction}")
    print(f" native height   : {h}")
    print(f" source native   : {src_w}x{h}")
    print(f" new core width  : {add_width_native}")
    print(f" overlap width   : {overlap_width}")
    print(f" context width   : {context_width}")
    print(f" canvas size     : {total_w}x{h}")
    print(f" canvas          : {generation_canvas_path}")
    print(f" mask            : {generation_mask_path}")
    print(f" prompt          : {prompt_path}")
    print(f" source native   : {source_native_path}")
    print(f" job             : {job_dir / 'job.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare a SHARED OVERLAP expansion job for Ravenfall-style world extension.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Create generation canvas/mask and job.json")
    p_prepare.add_argument("--source", required=True, help="Path to the existing source region PNG")
    p_prepare.add_argument("--job", required=True, help="Output job directory")
    p_prepare.add_argument("--direction", choices=["left", "right"], required=True)
    p_prepare.add_argument("--add-width", type=int, required=True, help="Width of the truly new area in native-height space")
    p_prepare.add_argument("--native-height", type=int, default=DEFAULT_NATIVE_HEIGHT)
    p_prepare.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p_prepare.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        prepare_job(
            source_path=Path(args.source),
            job_dir=Path(args.job),
            direction=args.direction,
            add_width_native=args.add_width,
            native_height=args.native_height,
            overlap_width=args.overlap,
            context_width=args.context,
        )
