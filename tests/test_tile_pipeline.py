import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from prepare_region_grid import prepare
from stitch_region_grid import stitch
from build_grid_from_jobs import build
from tile_grid import load_json, save_json


class TilePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.jobs = Path(self.temp.name) / 'jobs'
        (self.jobs / 'center').mkdir(parents=True)
        self.w, self.h, self.ov = 48, 36, 8
        Image.fromarray(self.patch(0, 0)).save(self.jobs / 'center' / 'region.png')

    def patch(self, x, y, width=None, height=None):
        xx, yy = np.meshgrid(np.arange(width or self.w) + x,
                             np.arange(height or self.h) + y)
        return np.stack([(xx * 3 + yy) % 256, (yy * 5 + xx) % 256,
                         (xx + yy * 2) % 256], axis=2).astype(np.uint8)

    def add(self, name, x, y, side):
        job = self.jobs / name
        prepare(self.jobs, job, x, y, side, overlap=self.ov, context=12)
        meta = load_json(job / 'grid_job.json')
        px, py = x * (self.w - self.ov), y * (self.h - self.ov)
        ew, eh = meta['expected_generation_size']
        generated = self.patch(px, py, ew, eh)
        Image.fromarray(generated).save(job / 'generated.png')
        stitch(job, job / 'generated.png', feather=0)
        return job

    def test_chain_and_all_directions_reconstruct_world_without_panorama_inputs(self):
        self.add('west_01', -1, 0, 'east')
        self.add('west_02', -2, 0, 'east')
        self.add('east_01', 1, 0, 'west')
        self.add('south_west', -2, 1, 'north')
        self.add('south_middle', -1, 1, 'west')  # north + west neighbors
        self.add('north', 0, -1, 'south')
        self.add('south_deep', -2, 2, 'north')
        result = build(self.jobs)
        with Image.open(result) as im:
            output = np.array(im)
        self.assertEqual(im.size, (168, 120))
        # Every occupied pixel must equal the same continuous synthetic map.
        expected = self.patch(-80, -28, 168, 120)
        occupied = output[:, :, 3] != 0
        difference = np.abs(output[:, :, :3][occupied].astype(np.int16)
                            - expected[occupied].astype(np.int16))
        # The compositor blends identical source pixels in float and rounds to
        # uint8; that can lower a few values by one without moving geography.
        self.assertLessEqual(int(difference.max()), 1)
        self.assertLess(float(np.count_nonzero(difference) / difference.size), .01)
        self.assertTrue((~occupied).any())
        self.assertEqual(len(load_json(result.parent / 'grid.json')['regions']), 8)
        for job in self.jobs.iterdir():
            if (job / 'tile.png').exists():
                with Image.open(job / 'tile.png') as tile:
                    self.assertEqual(tile.size, (48, 36))
                self.assertFalse((job / 'stitched' / 'final_native.png').exists())
        # Export can be deleted; next preparation still works from tiles alone.
        for f in (self.jobs / '_grid').iterdir():
            f.unlink()
        self.add('west_03', -3, 0, 'east')

    def test_context_has_correct_world_offset(self):
        for name, x, side in [('left', -1, 'east'), ('right', 1, 'west')]:
            job = self.jobs / name
            prepare(self.jobs, job, x, 0, side, overlap=8, context=12)
            canvas = np.array(Image.open(job / 'input' / 'generation_canvas.png'))
            source = self.patch(0, 0)
            context = np.array(Image.open(job / 'input' / 'primary_context.png'))
            if side == 'east':
                np.testing.assert_array_equal(context, source[:, :20])
                np.testing.assert_array_equal(canvas[:, -8:], source[:, :8])
            else:
                np.testing.assert_array_equal(context, source[:, -20:])
                np.testing.assert_array_equal(canvas[:, :8], source[:, -8:])

    def test_primary_overlap_is_fixed_on_canvas_and_in_tile(self):
        for name, x, y, side in [('left', -1, 0, 'east'), ('right', 1, 0, 'west'),
                                  ('below', 0, 1, 'north'), ('above', 0, -1, 'south')]:
            job = self.add(name, x, y, side)
            canvas = np.array(Image.open(job / 'input/generation_canvas.png'))
            mask = np.array(Image.open(job / 'input/generation_mask.png'))
            tile = np.array(Image.open(job / 'tile.png'))
            src = self.patch(0, 0)
            if side == 'east':
                a, b, c, fixed = canvas[:, -8:], tile[:, -2:], src[:, :8], mask[:, -8:]
            elif side == 'west':
                a, b, c, fixed = canvas[:, :8], tile[:, :2], src[:, -8:], mask[:, :8]
            elif side == 'north':
                a, b, c, fixed = canvas[:8], tile[:2], src[-8:], mask[:8]
            else:
                a, b, c, fixed = canvas[-8:], tile[-2:], src[:8], mask[-8:]
            np.testing.assert_array_equal(a, c)
            if side == 'east':
                np.testing.assert_array_equal(b, c[:, -2:])
            elif side == 'west':
                np.testing.assert_array_equal(b, c[:, :2])
            elif side == 'north':
                np.testing.assert_array_equal(b, c[:2])
            else:
                np.testing.assert_array_equal(b, c[-2:])
            self.assertTrue((fixed == 0).all())

    def test_changed_anchor_rejected_without_overwriting_existing_tile(self):
        job = self.add('west', -1, 0, 'east')
        before = (job / 'tile.png').read_bytes()
        altered = self.patch(-40, 0)
        altered[:, -8:] = 255 - altered[:, -8:]
        Image.fromarray(altered).save(job / 'generated.png')
        meta = load_json(job / 'grid_job.json')
        meta.pop('workflow', None)
        save_json(job / 'grid_job.json', meta)
        with self.assertRaisesRegex(ValueError, 'Terrain could not be aligned'):
            stitch(job, job / 'generated.png')
        self.assertEqual((job / 'tile.png').read_bytes(), before)

    def test_drifted_draft_does_not_publish_or_overwrite_tile(self):
        job = self.add('west', -1, 0, 'east')
        before_tile = (job / 'tile.png').read_bytes()
        before_meta = (job / 'region.json').read_bytes()
        changed = self.patch(-40, 0)
        changed[:, -8:] = 255 - changed[:, -8:]
        Image.fromarray(changed).save(job / 'generated.png')
        meta = load_json(job / 'grid_job.json')
        meta.pop('workflow', None)
        save_json(job / 'grid_job.json', meta)
        stitch(job, job / 'generated.png', draft=True)
        self.assertEqual((job / 'tile.png').read_bytes(), before_tile)
        self.assertEqual((job / 'region.json').read_bytes(), before_meta)
        self.assertTrue((job / 'draft/primary_pair_preview.png').exists())
        self.assertTrue(load_json(job / 'draft/stitch_meta.json')['anchor_changed'])
        (job / 'tile.png').unlink()
        (job / 'region.json').unlink()
        stitch(job, job / 'generated.png', draft=True)
        self.assertFalse((job / 'tile.png').exists())
        self.assertFalse((job / 'region.json').exists())

    def test_bad_aspect_ratio_does_not_publish_tile(self):
        job = self.jobs / 'west'
        prepare(self.jobs, job, -1, 0, 'east', overlap=8, context=12)
        Image.new('RGB', (48, 30)).save(job / 'generated.png')
        with self.assertRaisesRegex(ValueError, 'aspect ratio'):
            stitch(job, job / 'generated.png')
        self.assertFalse((job / 'region.json').exists())
        self.assertFalse((job / 'tile.png').exists())

    def test_scaled_generation_and_metadata(self):
        job = self.jobs / 'west'
        prepare(self.jobs, job, -1, 0, 'east', overlap=8, context=12)
        Image.fromarray(self.patch(-40, 0)).resize((96, 72), Image.Resampling.LANCZOS).save(job / 'generated.png')
        stitch(job, job / 'generated.png')
        with Image.open(job / 'tile.png') as im:
            self.assertEqual(im.size, (48, 36))
        meta = load_json(job / 'region.json')
        self.assertEqual(meta['image'], 'tile.png')
        self.assertEqual(meta['dependencies'], ['center'])

    def test_existing_job_and_inconsistent_geometry_rejected(self):
        job = self.add('west', -1, 0, 'east')
        with self.assertRaises(FileExistsError):
            prepare(self.jobs, job, -2, 0, 'east')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            prepare(self.jobs, self.jobs / 'next', -2, 0, 'east', overlap=9)

    def test_500_tiles_stay_local(self):
        layout = {'version': 1, 'width': 48, 'height': 36, 'overlap': 8}
        save_json(self.jobs / 'layout.json', layout)
        previous = 'center'
        for i in range(1, 501):
            name = f'west_{i:03}'
            job = self.jobs / name
            job.mkdir()
            Image.new('RGB', (48, 36), (i % 256, 20, 30)).save(job / 'tile.png')
            save_json(job / 'region.json', {
                'format': 'local_tile_v1', 'id': name, 'x': -i, 'y': 0,
                'image': 'tile.png', 'layout': layout, 'dependencies': [previous],
            })
            previous = name
        result = build(self.jobs)
        with Image.open(result) as im:
            self.assertEqual(im.size, (20048, 36))
            self.assertEqual(im.getpixel((0, 0)), (500 % 256, 20, 30, 255))
        self.assertEqual(len(load_json(result.parent / 'grid.json')['regions']), 501)
        self.assertEqual(len(list((self.jobs / 'west_500').iterdir())), 2)

    def test_cli_end_to_end(self):
        root = Path(__file__).resolve().parents[1]
        def run(script, *args):
            subprocess.run([sys.executable, '-B', str(root / script), *map(str, args)],
                           cwd=root, check=True, capture_output=True, text=True)
        job = self.jobs / 'west'
        run('prepare_region_grid.py', '--jobs', self.jobs, '--job', job,
            '--x', '-1', '--y', '0', '--primary', 'east', '--overlap', '8', '--context', '12')
        Image.fromarray(self.patch(-40, 0)).save(job / 'generated.png')
        run('stitch_region_grid.py', '--job', job, '--generated', job / 'generated.png')
        run('build_grid_from_jobs.py', '--jobs', self.jobs)
        manifest = load_json(self.jobs / '_grid' / 'grid.json')
        self.assertEqual({r['id'] for r in manifest['regions']}, {'center', 'west'})
        self.assertTrue((job / 'tile.png').exists())
        self.assertFalse((job / 'stitched' / 'final_native.png').exists())

    def test_old_panorama_is_ignored_and_build_is_repeatable(self):
        job = self.add('west', -1, 0, 'east')
        (job / 'stitched').mkdir()
        Image.new('RGB', (999, 20), 'red').save(job / 'stitched' / 'final_native.png')
        path = build(self.jobs)
        first = path.read_bytes()
        build(self.jobs)
        self.assertEqual(first, path.read_bytes())
        with Image.open(path) as im:
            self.assertEqual(im.size, (88, 36))


if __name__ == '__main__':
    unittest.main()
