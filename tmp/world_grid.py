#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

DIRS = {
    'north': (0, -1),
    'east': (1, 0),
    'south': (0, 1),
    'west': (-1, 0),
}


def load_world(path: Path) -> dict:
    if not path.exists():
        return {'version': 1, 'regions': {}}
    return json.loads(path.read_text(encoding='utf-8'))


def save_world(path: Path, world: dict) -> None:
    path.write_text(json.dumps(world, ensure_ascii=False, indent=2), encoding='utf-8')


def key(x: int, y: int) -> str:
    return f'{x},{y}'


def neighbors(world: dict, x: int, y: int) -> dict:
    out = {}
    for side, (dx, dy) in DIRS.items():
        r = world.get('regions', {}).get(key(x + dx, y + dy))
        if r:
            out[side] = r
    return out


def cmd_init(args):
    p = Path(args.world)
    if p.exists() and not args.force:
        raise FileExistsError(f'{p} already exists; use --force')
    save_world(p, {'version': 1, 'regions': {}})
    print(p)


def cmd_add(args):
    p = Path(args.world)
    world = load_world(p)
    img = str(Path(args.image).resolve())
    region = {
        'id': args.id,
        'x': args.x,
        'y': args.y,
        'image': img,
    }
    if args.meta:
        region['meta'] = str(Path(args.meta).resolve())
    world.setdefault('regions', {})[key(args.x, args.y)] = region
    save_world(p, world)
    print(f"registered {args.id} at ({args.x},{args.y}) -> {img}")


def cmd_neighbors(args):
    world = load_world(Path(args.world))
    print(json.dumps(neighbors(world, args.x, args.y), ensure_ascii=False, indent=2))


def cmd_list(args):
    world = load_world(Path(args.world))
    rows = sorted(world.get('regions', {}).values(), key=lambda r: (r['y'], r['x']))
    for r in rows:
        print(f"({r['x']:>3},{r['y']:>3})  {r['id']:<24} {r['image']}")


def parser():
    p = argparse.ArgumentParser(description='Region-grid manifest for The Frontier world map')
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('init'); a.add_argument('--world', default='world.json'); a.add_argument('--force', action='store_true'); a.set_defaults(func=cmd_init)
    a = sub.add_parser('add'); a.add_argument('--world', default='world.json'); a.add_argument('--id', required=True); a.add_argument('--x', type=int, required=True); a.add_argument('--y', type=int, required=True); a.add_argument('--image', required=True); a.add_argument('--meta'); a.set_defaults(func=cmd_add)
    a = sub.add_parser('neighbors'); a.add_argument('--world', default='world.json'); a.add_argument('--x', type=int, required=True); a.add_argument('--y', type=int, required=True); a.set_defaults(func=cmd_neighbors)
    a = sub.add_parser('list'); a.add_argument('--world', default='world.json'); a.set_defaults(func=cmd_list)
    return p

if __name__ == '__main__':
    args = parser().parse_args(); args.func(args)
