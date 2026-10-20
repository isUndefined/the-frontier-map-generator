"""Incremental, art-first world generation CLI.

This wraps the existing local-tile preparation and stitching pipeline; it does
not create a cumulative panorama or pretend to be an image generation model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone
from terrain_quality import audit_tile
import sys
import zipfile

from PIL import Image

from prepare_region_grid import prepare
from stitch_region_grid import stitch
from tile_grid import DIRS, discover, geometry, load_json, save_json, ordered_regions

OPPOSITE = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}


def occupied_and_pending(jobs: Path):
    occupied = discover(jobs)
    pending = {}
    for job in jobs.iterdir():
        meta_path = job / 'grid_job.json'
        if not job.is_dir() or not meta_path.is_file() or (job / 'region.json').is_file():
            continue
        meta = load_json(meta_path)
        if meta.get('method') != 'local_tile_v1':
            continue
        target = meta.get('target', {})
        pos = (int(target['x']), int(target['y']))
        if pos in pending or pos in occupied:
            raise ValueError(f'Duplicate pending/occupied position {pos}: {job}')
        pending[pos] = job.name
    return occupied, pending


def find_anchor(regions, spec: str):
    spec = spec.strip()
    if ',' in spec:
        try:
            x, y = (int(v.strip()) for v in spec.split(','))
        except ValueError as exc:
            raise ValueError('--from must be tile ID or X,Y') from exc
        if (x, y) not in regions:
            raise ValueError(f'No published tile at ({x}, {y})')
        return (x, y)
    matches = [(x, y) for (x, y), v in regions.items() if v['id'] == spec or v['job'] == spec]
    if len(matches) != 1:
        raise ValueError(f'Unknown or ambiguous tile ID: {spec}')
    return matches[0]


def choose_target(regions, direction, anchor_id=None):
    if not regions:
        raise ValueError('First create jobs/center/region.png')
    dx, dy = DIRS[direction]
    if anchor_id is not None:
        x, y = find_anchor(regions, anchor_id)
    else:
        # Extend the directional frontier, preferring the row/column closest
        # to the origin to avoid the CLI jumping unpredictably between arms.
        candidates = [(x, y) for x, y in regions if (x + dx, y + dy) not in regions]
        if not candidates:
            raise ValueError(f'No available frontier in direction {direction}')
        if direction == 'west':
            key = lambda p: (p[0], abs(p[1]), p[1])
        elif direction == 'east':
            key = lambda p: (-p[0], abs(p[1]), p[1])
        elif direction == 'north':
            key = lambda p: (p[1], abs(p[0]), p[0])
        else:
            key = lambda p: (-p[1], abs(p[0]), p[0])
        x, y = min(candidates, key=key)
    target = (x + dx, y + dy)
    if target in regions:
        raise ValueError(f'Tile at {target} already exists; choose another --from tile')
    return (x, y), target


def default_job_name(pos):
    x, y = pos
    return f'tile_x{x:+d}_y{y:+d}'


def prepare_expansion(jobs, direction, anchor=None, name=None, context=700,
                      overlap=None, dry_run=False):
    jobs = Path(jobs).resolve()
    if not jobs.is_dir():
        raise ValueError(f'Jobs directory does not exist: {jobs}')
    layout = geometry(jobs, overlap=overlap)
    regions, pending = occupied_and_pending(jobs)
    source, target = choose_target(regions, direction, anchor)
    if target in pending:
        raise ValueError(f'Tile at {target} is already being prepared as {pending[target]}; finalize it first')
    name = name or default_job_name(target)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.+-]{0,99}', name) or name in {'center', 'jobs'} or name.startswith('_'):
        raise ValueError('Invalid job name; use letters, digits, underscore, dot, plus or minus')
    path = jobs / name
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Job directory already exists: {path}')
    data = {
        'job': name, 'source': {'x': source[0], 'y': source[1], 'id': regions[source]['id']},
        'target': {'x': target[0], 'y': target[1]},
        'primary_side': OPPOSITE[direction], 'direction': direction,
        'jobs': str(jobs), 'layout': layout,
        'generation': 'not_started' if dry_run else 'awaiting_image',
    }
    if not dry_run:
        prepare(jobs, path, *target, OPPOSITE[direction], context=context, overlap=overlap)
        data['inputs'] = {
            'canvas': str(path / 'input' / 'generation_canvas.png'),
            'mask': str(path / 'input' / 'generation_mask.png'),
            'prompt': str(path / 'input' / 'prompt.txt'),
            'context': str(path / 'input' / 'context_board.png'),
            'debug': str(path / 'input' / 'neighbor_debug_preview.png'),
        }
        data['generated'] = str(path / 'generated.png')
    return data



def rebuild_job(jobs: Path, job: str, context=None):
    """Safely replace job inputs while leaving published terrain untouched."""
    jobs = Path(jobs).resolve()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.+-]{0,99}', job) or job in {'center', 'jobs'} or job.startswith('_'):
        raise ValueError('Invalid job name')
    folder = jobs / job
    old_meta_path = folder / 'grid_job.json'
    published_meta_path = folder / 'region.json'
    if not old_meta_path.is_file() or not published_meta_path.is_file():
        raise ValueError('rebuild-job requires a published region and grid_job.json')
    old = load_json(old_meta_path)
    published = load_json(published_meta_path)
    x, y = int(old['target']['x']), int(old['target']['y'])
    if (int(published['x']), int(published['y'])) != (x, y) or published['id'] != old['region_id']:
        raise ValueError('Published region coordinates/id differ from prepared job')
    regs = discover(jobs)
    if regs.get((x, y), {}).get('id') != published['id']:
        raise ValueError('Published tile is not registered at the expected coordinates')
    side = old['primary_side']
    if side not in DIRS or (x + DIRS[side][0], y + DIRS[side][1]) not in regs:
        raise ValueError(f'Existing primary neighbor {side} is unavailable')
    ctx = int(context if context is not None else old.get('context', 700))
    if ctx < 0:
        raise ValueError('Context must be nonnegative')

    stage = Path(tempfile.mkdtemp(prefix='rebuild_stage_', dir=jobs))
    moved = []
    try:
        prepare(jobs, stage, x, y, side, region_id=published['id'],
                context=ctx, allow_occupied_id=published['id'])
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = folder / 'backups' / ('input_' + stamp)
        backup.mkdir(parents=True, exist_ok=False)
        for name in ('input', 'grid_job.json', 'generated.png', 'draft', 'quality_report.json'):
            src = folder / name
            if src.exists():
                dest = backup / name
                src.rename(dest)
                moved.append((src, dest))
        for name in ('input', 'grid_job.json'):
            (stage / name).rename(folder / name)
        return {'job': job, 'target': {'x': x, 'y': y},
                'backup': str(backup),
                'canvas': str(folder / 'input/generation_canvas.png'),
                'mask': str(folder / 'input/generation_mask.png'),
                'prompt': str(folder / 'input/prompt.txt'),
                'debug': str(folder / 'input/neighbor_debug_preview.png'),
                'published_unchanged': str(folder / 'tile.png')}
    except Exception:
        old_names = {src.name for src, _ in moved}
        for name in ('input', 'grid_job.json'):
            candidate = folder / name
            if name in old_names and candidate.exists():
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
        for src, dest in reversed(moved):
            if dest.exists():
                dest.rename(src)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def pack_chatgpt(jobs: Path, job: str) -> Path:
    """Make a reusable ChatGPT handoff ZIP for an already prepared job.

    Includes every prepared source/reference image and prompt, but no generated
    output, published tiles, private host paths or accumulated world panoramas.
    Does not modify existing input files; safe to re-run after a failed attempt.
    """
    jobs = Path(jobs).resolve()
    folder = jobs / job
    meta = folder / 'grid_job.json'
    inp = folder / 'input'
    if folder.parent != jobs or not meta.is_file() or not inp.is_dir():
        raise ValueError(f'No prepared job under jobs/: {job}')
    if (folder / 'region.json').is_file():
        raise ValueError(f'Job {job} is already published')
    metadata = load_json(meta)
    if metadata.get('method') != 'local_tile_v1':
        raise ValueError(f'Unsupported job format: {job}')
    wanted = ['generation_canvas.png', 'generation_mask.png', 'prompt.txt', 'context_board.png']
    missing = [x for x in wanted if not (inp / x).is_file()]
    if missing:
        raise FileNotFoundError(f'Job {job} is missing inputs: {", ".join(missing)}')
    with Image.open(inp / 'generation_canvas.png') as im:
        width, height = im.size
    with Image.open(inp / 'generation_mask.png') as im:
        if im.size != (width, height):
            raise ValueError('Generation mask must match canvas dimensions')
    bundle_path = folder / 'chatgpt_bundle.zip'
    prompt = (inp / 'prompt.txt').read_text(encoding='utf-8')
    guide = (
        f'# The Frontier / ChatGPT art continuation — {job}\n\n'
        f'Target tile: {metadata.get("target", {})}\n'
        f'Output filename: generated.png\n'
        f'OUTPUT SIZE: EXACTLY {width} x {height} pixels.\n'
        'Do not resize, upscale, downscale, stretch, crop, rotate, or reframe the canvas.\n'
        'Preserve pixels in BLACK areas of input/generation_mask.png exactly.\n'
        'Generate new terrain ONLY where the mask is WHITE.\n'
        'Use input/context_board.png and input/neighbors/ / input/diagonals/ '
        'as visual references, if present.\n'
        'Continue roads, streams, mountains, lighting and perspective seamlessly.\n'
        'Return ONE complete PNG canvas, with no labels or interface.\n\n'
        '--- ORIGINAL GENERATION PROMPT ---\n\n' + prompt + '\n'
    )
    tmp = bundle_path.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            z.writestr('CHATGPT_INSTRUCTIONS.md', guide)
            for image in sorted(inp.rglob('*')):
                if image.is_file() and not image.is_symlink():
                    relative = image.relative_to(folder).as_posix()
                    z.write(image, relative)
            # Avoid absolute container paths found in grid_job.json; the
            # prompt and prepared sources are sufficient for the conversation.
            z.writestr('job_info.json', json.dumps({
                'job': job, 'target': metadata.get('target'),
                'primary_side': metadata.get('primary_side'),
                'canvas_size': [width, height],
            }, ensure_ascii=False, indent=2))
        tmp.replace(bundle_path)
    finally:
        tmp.unlink(missing_ok=True)
    return bundle_path



def publish(jobs: Path, job: str, generated=None, feather=4, draft=False,
            export=True, export_format='webp', replace=False, allow_weak_alignment=False):
    """Validate a draft before publishing; preserve old assets on replacement.

    The quality gate runs on the *stitched* draft and never changes world
    registrations on rejection. Replacements are backed up before modifying
    existing assets, and failed writes restore the previous publication.
    """
    jobs = Path(jobs).resolve()
    path = jobs / job
    if not path.is_dir() or not (path / 'grid_job.json').exists():
        raise ValueError(f'No prepared job: {job}')
    published = (path / 'region.json').exists()
    if published and not replace and not draft:
        raise ValueError(f'Job {job} is already published; use --replace after inspection')
    img = Path(generated).resolve() if generated else path / 'generated.png'
    if not img.is_file():
        raise FileNotFoundError(f'Missing image {img}. Save a new PNG then run finalize.')
    with Image.open(path / 'input' / 'generation_canvas.png') as canvas:
        expected_size = canvas.size
    with Image.open(img) as result:
        actual_size = result.size
    if actual_size != expected_size:
        raise ValueError(
            f'Generated image is {actual_size}, expected {expected_size}. '
            'No resize or upscaling permitted.'
        )

    # Always use the original stitcher draft path first. It cannot overwrite
    # the registered tile or its published metadata.
    stitch(path, img, feather=feather, draft=True, jobs=jobs)
    draft_path = path / 'draft' / 'tile.png'
    report = audit_tile(
        jobs, job, draft_path,
        stitch_metadata=load_json(path / 'draft' / 'stitch_meta.json'),
        allow_weak_alignment=allow_weak_alignment,
    )
    save_json(path / 'draft' / 'quality_report.json', report)
    if draft:
        return {'job': job, 'draft': str(draft_path),
                'world_preview': str(path / 'draft' / 'world_preview.png'),
                'world_preview_report': str(path / 'draft' / 'world_preview.json'),
                'quality_report': str(path / 'draft' / 'quality_report.json'),
                'quality': report['status'], 'issues': report['issues']}
    if report['status'] != 'passed':
        raise ValueError(
            f'Quality gate rejected {job}: {", ".join(report["issues"])}. '
            'View draft/quality_report.json, regenerate the artwork, or '
            'use --allow-weak-alignment ONLY for weak alignment. '
            'Duplicated terrain is never overridable.'
        )

    # A validated draft precedes publication, so weak/no-registration fallback
    # never silently sneaks into the world.
    backups = {}
    if replace:
        if not published:
            raise ValueError('--replace is for a previously published job only')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup_dir = path / 'backups' / stamp
        backup_dir.mkdir(parents=True, exist_ok=False)
        for filename in ('tile.png', 'region.json', 'stitch_meta.json'):
            target = path / filename
            if target.is_file():
                shutil.copy2(target, backup_dir / filename)
                backups[filename] = backup_dir / filename
    else:
        backup_dir = None

    try:
        # Stitch a final tile with correct dependencies. The draft was checked
        # independently; recheck again if its bytes ever change unexpectedly.
        stitch(path, img, feather=feather, draft=False, jobs=jobs)
        output = path / 'tile.png'
        final_report = audit_tile(
            jobs, job, output,
            stitch_metadata=load_json(path / 'stitch_meta.json'),
            allow_weak_alignment=allow_weak_alignment,
        )
        save_json(path / 'quality_report.json', final_report)
        if final_report['status'] != 'passed':
            raise ValueError(f'Final tile failed QA: {final_report["issues"]}')
        if export:
            manifest = export_tiles(jobs, export_format, force=replace)
    except Exception:
        if replace:
            for filename in ('tile.png', 'region.json', 'stitch_meta.json'):
                p = path / filename
                if filename in backups:
                    shutil.copy2(backups[filename], p)
                else:
                    p.unlink(missing_ok=True)
            # Refresh stale browser tiles if possible.
            try:
                if export:
                    export_tiles(jobs, export_format, force=True)
            except Exception:
                pass
        else:
            for filename in ('tile.png', 'region.json', 'stitch_meta.json', 'quality_report.json'):
                (path / filename).unlink(missing_ok=True)
        raise

    result = {'job': job, 'published': str(path / 'tile.png'),
              'quality_report': str(path / 'quality_report.json'),
              'quality': 'passed'}
    if backup_dir is not None:
        result['backup'] = str(backup_dir)
    if export:
        result['manifest'] = str(manifest)
        result['viewer'] = str(jobs / '_export' / 'index.html')
    return result


def export_tiles(jobs: Path, output_format='webp', quality=92, force=False, lossy=False):
    if output_format not in ('webp', 'png'):
        raise ValueError('Choose --format webp or png')
    jobs = Path(jobs).resolve()
    layout = geometry(jobs)
    regions = discover(jobs)
    if not regions:
        raise ValueError('No published tiles found')
    output = jobs / '_export'
    tiles_dir = output / 'tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)
    step_x, step_y = layout['width'] - layout['overlap'], layout['height'] - layout['overlap']
    entries = []
    # Paint dependency tiles first so their exact overlap can be replaced by
    # the newer tile. Positional sorting would break multi-neighbor corners.
    for region in ordered_regions(regions):
        x, y = int(region['x']), int(region['y'])
        source = Path(region['image'])
        filename = f'x{x:+d}_y{y:+d}.{output_format}'
        destination = tiles_dir / filename
        if force or not destination.is_file() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            with Image.open(source) as im:
                if im.size != (layout['width'], layout['height']):
                    raise ValueError(f'Invalid tile dimensions in {source}: {im.size}')
                temp = destination.with_name(destination.name + '.tmp')
                # Encode losslessly by default: lossy compression applied
                # independently to neighboring tiles can reintroduce seams.
                kwargs = ({'quality': quality, 'method': 6, 'lossless': not lossy}
                          if output_format == 'webp' else {})
                im.convert('RGB').save(temp, format=output_format.upper(), **kwargs)
                temp.replace(destination)
        entries.append({
            'id': region['id'], 'x': x, 'y': y,
            'pixel_x': x * step_x, 'pixel_y': y * step_y,
            'width': layout['width'], 'height': layout['height'],
            'image': f'tiles/{filename}',
            'dependencies': region.get('dependencies', []),
        })
    manifest = {
        'format': 'frontier_art_tiles_v1', 'coordinate_system': 'grid_y_down',
        'layout': layout, 'step_px': {'x': step_x, 'y': step_y},
        'render_order': 'dependencies_first', 'tiles': entries,
    }
    # The tiles array is in dependency-first paint order.
    save_json(output / 'manifest.json', manifest)
    # Always publish a lightweight viewer alongside the manifest. This keeps
    # browsing independent from the number and size of world tiles.
    viewer_source = Path(__file__).resolve().parent / 'viewer' / 'index.html'
    if not viewer_source.is_file():
        raise FileNotFoundError(f'Missing viewer source: {viewer_source}')
    shutil.copyfile(viewer_source, output / 'index.html')
    return output / 'manifest.json'


def show_status(jobs):
    jobs = Path(jobs).resolve()
    layout = geometry(jobs)
    regions, pending = occupied_and_pending(jobs)
    return {'jobs': str(jobs), 'layout': layout,
            'published': [{'id': v['id'], 'x': x, 'y': y}
                          for (x, y), v in sorted(regions.items())],
            'pending': [{'job': job, 'x': x, 'y': y}
                        for (x, y), job in sorted(pending.items())]}


def build_parser():
    p = argparse.ArgumentParser(description='Expand The Frontier world with manual canvas/mask/prompt.')
    p.add_argument('--jobs', default='jobs', help='Jobs directory; Docker mounts it as /app/jobs')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('status', help='List published and pending tiles')
    expand = sub.add_parser('expand', help='Prepare canvas, mask and prompt for a new tile')
    expand.add_argument('--direction', choices=DIRS, required=True)
    expand.add_argument('--provider', choices=['chatgpt'], default='chatgpt')
    expand.add_argument('--from', dest='anchor', help='Published tile ID or grid coordinates x,y')
    expand.add_argument('--job', help='Name of the new job')
    expand.add_argument('--context', type=int, default=700)
    expand.add_argument('--overlap', type=int, default=None)
    expand.add_argument('--dry-run', action='store_true')
    rebuild = sub.add_parser('rebuild-job', help='Refresh canvas/mask/prompt using all overlapping neighbors')
    rebuild.add_argument('--job', required=True)
    rebuild.add_argument('--context', type=int, default=None)
    pack = sub.add_parser('pack', help='Create/re-create a ChatGPT ZIP for an existing pending job')
    pack.add_argument('--job', required=True)
    finalize = sub.add_parser('finalize', help='Validate and publish a ChatGPT-generated PNG')
    finalize.add_argument('--job', required=True)
    finalize.add_argument('--generated', help='PNG path visible inside Docker (e.g. /app/jobs/<job>/generated.png)')
    finalize.add_argument('--feather', type=int, default=4)
    finalize.add_argument('--draft', action='store_true')
    finalize.add_argument('--no-export', action='store_true')
    finalize.add_argument('--replace', action='store_true',
                          help='Replace a published tile after QA, backing up old art and metadata')
    finalize.add_argument('--allow-weak-alignment', action='store_true',
                          help='Allow failed alignment only; duplicate terrain remains a hard blocker')
    audit = sub.add_parser('audit', help='Quality-check an existing published tile')
    audit.add_argument('--job', required=True)
    audit.add_argument('--allow-weak-alignment', action='store_true')
    ex = sub.add_parser('export', help='Export separate browser tiles and manifest, never a panorama')
    ex.add_argument('--format', choices=('webp', 'png'), default='webp')
    ex.add_argument('--quality', type=int, default=92)
    ex.add_argument('--force', action='store_true')
    ex.add_argument('--lossy', action='store_true')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        jobs = Path(args.jobs).resolve()
        if args.command == 'status':
            result = show_status(jobs)
        elif args.command == 'export':
            result = {'manifest': str(export_tiles(jobs, args.format, args.quality, args.force, args.lossy)),
                      'viewer': str(jobs / '_export' / 'index.html')}
        elif args.command == 'rebuild-job':
            result = rebuild_job(jobs, args.job, context=args.context)
        elif args.command == 'pack':
            bundle = pack_chatgpt(jobs, args.job)
            result = {'job': args.job, 'chatgpt_bundle': str(bundle),
                      'next_step': f'Upload this ZIP to ChatGPT. Save the resulting PNG as jobs/{args.job}/generated.png, then run finalize --job {args.job}.'}
        elif args.command == 'audit':
            tile = jobs / args.job / 'tile.png'
            if not (jobs / args.job / 'region.json').is_file():
                raise ValueError(f'Job {args.job} is not published')
            result = audit_tile(jobs, args.job, tile,
                                allow_weak_alignment=args.allow_weak_alignment)
            save_json(jobs / args.job / 'quality_report.json', result)
        elif args.command == 'finalize':
            result = publish(jobs, args.job, generated=args.generated, feather=args.feather,
                             draft=args.draft, export=not args.no_export,
                             replace=args.replace, allow_weak_alignment=args.allow_weak_alignment)
        else:
            result = prepare_expansion(jobs, args.direction, args.anchor, args.job,
                                       args.context, args.overlap, args.dry_run)
            if not args.dry_run:
                result['next_step'] = (
                    'Send input/generation_canvas.png, input/generation_mask.png '
                    'and input/prompt.txt to ChatGPT; save its PNG as generated.png '
                    'and run finalize.'
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
