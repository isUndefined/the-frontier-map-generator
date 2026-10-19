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
import shlex
import subprocess
import sys

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
        }
        data['generated'] = str(path / 'generated.png')
    return data


def run_generator(command: str, jobs: Path, job: str, target):
    if not command.strip():
        raise ValueError('Image generation command is empty')
    path = jobs / job
    output = path / 'generated.png'
    env = {**os.environ,
           'FRONTIER_CANVAS': str(path / 'input' / 'generation_canvas.png'),
           'FRONTIER_MASK': str(path / 'input' / 'generation_mask.png'),
           'FRONTIER_PROMPT': str(path / 'input' / 'prompt.txt'),
           'FRONTIER_CONTEXT': str(path / 'input' / 'context_board.png'),
           'FRONTIER_OUTPUT': str(output),
           'FRONTIER_JOB': str(path),
           'FRONTIER_TARGET_X': str(target['x']),
           'FRONTIER_TARGET_Y': str(target['y'])}
    # Do not use shell=True: paths and template contents must not be executable.
    args = shlex.split(command)
    if not args:
        raise ValueError('Empty generator command')
    subprocess.run(args, env=env, check=True)
    if not output.is_file():
        raise FileNotFoundError(f'Generator exited without creating {output}')
    return output


def publish(jobs: Path, job: str, generated=None, feather=4, draft=False, export=True,
            export_format='webp'):
    jobs = Path(jobs).resolve()
    path = jobs / job
    if not path.is_dir() or not (path / 'grid_job.json').exists():
        raise ValueError(f'No prepared job: {job}')
    if (path / 'region.json').exists():
        raise ValueError(f'Job {job} is already published; refusing to overwrite')
    img = Path(generated).resolve() if generated else path / 'generated.png'
    if not img.is_file():
        raise FileNotFoundError(f'Missing image {img}; provide --generated or configure a model')
    stitch(path, img, feather=feather, draft=draft, jobs=jobs)
    if draft:
        return {'job': job, 'draft': str(path / 'draft' / 'tile.png')}
    if not (path / 'region.json').is_file():
        raise RuntimeError('Stitch returned without registering the tile')
    result = {'job': job, 'published': str(path / 'tile.png')}
    if export:
        result['manifest'] = str(export_tiles(jobs, export_format))
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
    p = argparse.ArgumentParser(description='Grow The Frontier world one art tile at a time.')
    p.add_argument('--jobs', default='jobs', help='Jobs directory; mounted as /app/jobs in Docker')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('status', help='List published and pending tiles')
    expand = sub.add_parser('expand', help='Prepare the next tile, optionally generate and publish')
    expand.add_argument('--direction', choices=DIRS, required=True)
    expand.add_argument('--from', dest='anchor', help='Starting published tile ID or coordinates x,y')
    expand.add_argument('--job', help='Name of the new job')
    expand.add_argument('--context', type=int, default=700)
    expand.add_argument('--overlap', type=int, default=None)
    expand.add_argument('--dry-run', action='store_true')
    expand.add_argument('--generator-command', default=None,
                        help='External generator command; may also use FRONTIER_GENERATOR_COMMAND')
    expand.add_argument('--generated', help='Existing image for this expansion')
    expand.add_argument('--feather', type=int, default=4)
    expand.add_argument('--no-export', action='store_true')
    finalize = sub.add_parser('finalize', help='Stitch and publish an existing prepared job')
    finalize.add_argument('--job', required=True)
    finalize.add_argument('--generated', help='External generated image')
    finalize.add_argument('--generator-command', default=None,
                          help='Run or retry an external generator before finalization')
    finalize.add_argument('--feather', type=int, default=4)
    finalize.add_argument('--draft', action='store_true')
    finalize.add_argument('--no-export', action='store_true')
    ex = sub.add_parser('export', help='Export separate browser tiles and manifest, never a panorama')
    ex.add_argument('--format', choices=('webp', 'png'), default='webp')
    ex.add_argument('--quality', type=int, default=92)
    ex.add_argument('--force', action='store_true')
    ex.add_argument('--lossy', action='store_true', help='Smaller WebP, but may introduce seam artifacts')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        jobs = Path(args.jobs).resolve()
        if args.command == 'status':
            result = show_status(jobs)
        elif args.command == 'export':
            result = {'manifest': str(export_tiles(jobs, args.format, args.quality, args.force, args.lossy))}
        elif args.command == 'finalize':
            cmd = args.generator_command or os.getenv('FRONTIER_GENERATOR_COMMAND', '')
            if args.generated and args.generator_command:
                raise ValueError('Choose --generated OR --generator-command')
            generation = None
            if args.generated:
                generation = Path(args.generated).resolve()
            elif cmd and not (jobs / args.job / 'generated.png').is_file():
                metadata = load_json(jobs / args.job / 'grid_job.json')
                generation = run_generator(cmd, jobs, args.job, metadata['target'])
            result = publish(jobs, args.job, generated=generation, feather=args.feather,
                             draft=args.draft, export=not args.no_export)
        else:
            if args.generated and args.generator_command:
                raise ValueError('Choose --generated OR --generator-command')
            result = prepare_expansion(jobs, args.direction, args.anchor, args.job,
                                       args.context, args.overlap, args.dry_run)
            if not args.dry_run:
                command = args.generator_command or os.getenv('FRONTIER_GENERATOR_COMMAND', '')
                if args.generated or command:
                    generated = (Path(args.generated).resolve() if args.generated
                                 else run_generator(command, jobs, result['job'], result['target']))
                    result.update(publish(jobs, result['job'], generated, args.feather,
                                          export=not args.no_export))
                else:
                    result['next_step'] = (
                        f'Create {result["generated"]} using input/generation_canvas.png, '
                        'input/generation_mask.png and input/prompt.txt; '
                        f'then: frontier_cli.py finalize --job {result["job"]}'
                    )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
