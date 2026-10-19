import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

import frontier_cli as cli
from tile_grid import load_json


class FrontierCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.jobs = Path(temp.name) / 'jobs'
        (self.jobs / 'center').mkdir(parents=True)
        self.w, self.h, self.ov = 54, 42, 8
        Image.fromarray(self.world_patch(0, 0)).save(self.jobs / 'center' / 'region.png')
        (self.jobs / 'layout.json').write_text(json.dumps({'version': 1, 'width': self.w, 'height': self.h, 'overlap': self.ov}))

    def world_patch(self, left, top):
        xx, yy = np.meshgrid(np.arange(self.w) + left, np.arange(self.h) + top)
        return np.stack([(xx + 3 * yy) % 256, (2 * xx + yy) % 256,
                         (xx + yy) % 256], axis=2).astype(np.uint8)

    def fake_output(self, plan):
        x, y = plan['target']['x'], plan['target']['y']
        path = self.jobs / plan['job'] / 'generated.png'
        Image.fromarray(self.world_patch(x * (self.w - self.ov),
                                         y * (self.h - self.ov))).save(path)
        return path

    def test_dry_run_does_not_write_files(self):
        data = cli.prepare_expansion(self.jobs, 'west', dry_run=True)
        self.assertEqual(data['target'], {'x': -1, 'y': 0})
        self.assertEqual(data['primary_side'], 'east')
        self.assertEqual(sorted(p.name for p in self.jobs.iterdir()), ['center', 'layout.json'])
        self.assertEqual(data['layout']['width'], self.w)

    def test_prepare_finalize_export_and_next_frontier(self):
        first = cli.prepare_expansion(self.jobs, 'west', context=12)
        self.assertEqual(len(cli.show_status(self.jobs)['pending']), 1)
        with self.assertRaisesRegex(ValueError, 'already being prepared'):
            cli.prepare_expansion(self.jobs, 'west')
        self.fake_output(first)
        report = cli.publish(self.jobs, first['job'], feather=0)
        self.assertTrue(Path(report['published']).is_file())
        self.assertEqual(len(cli.show_status(self.jobs)['pending']), 0)
        manifest = load_json(Path(report['manifest']))
        self.assertEqual(len(manifest['tiles']), 2)
        self.assertEqual(manifest['format'], 'frontier_art_tiles_v1')
        self.assertEqual(manifest['step_px'], {'x': 46, 'y': 34})
        self.assertEqual([t['id'] for t in manifest['tiles']], ['center', first['job']])
        exported = Path(report['manifest']).parent / manifest['tiles'][1]['image']
        self.assertTrue(exported.is_file())
        with Image.open(exported) as im, Image.open(self.jobs / first['job'] / 'tile.png') as source:
            np.testing.assert_array_equal(np.array(im.convert('RGB')), np.array(source.convert('RGB')))
        self.assertFalse((self.jobs / '_grid').exists())
        second = cli.prepare_expansion(self.jobs, 'west', context=12)
        self.assertEqual(second['target'], {'x': -2, 'y': 0})
        self.fake_output(second)
        cli.publish(self.jobs, second['job'], feather=0)
        self.assertEqual(len(load_json(self.jobs / '_export' / 'manifest.json')['tiles']), 3)

    def test_forced_anchor_and_occupied_guard(self):
        west = cli.prepare_expansion(self.jobs, 'west', context=12)
        self.fake_output(west)
        cli.publish(self.jobs, west['job'], feather=0)
        south = cli.prepare_expansion(self.jobs, 'south', anchor='-1,0', context=12)
        self.assertEqual(south['target'], {'x': -1, 'y': 1})
        with self.assertRaisesRegex(ValueError, 'already exists'):
            cli.prepare_expansion(self.jobs, 'west', anchor='center')

    def test_unknown_generator_does_not_publish(self):
        plan = cli.prepare_expansion(self.jobs, 'east', context=12)
        with self.assertRaises(FileNotFoundError):
            cli.publish(self.jobs, plan['job'])
        self.assertFalse((self.jobs / plan['job'] / 'region.json').exists())

    def test_chatgpt_zip_contents_and_existing_job_repack(self):
        plan = cli.prepare_expansion(self.jobs, 'east', context=12)
        bundle = cli.pack_chatgpt(self.jobs, plan['job'])
        self.assertTrue(bundle.is_file())
        with zipfile.ZipFile(bundle) as z:
            names = set(z.namelist())
            self.assertIn('CHATGPT_INSTRUCTIONS.md', names)
            self.assertIn('input/generation_canvas.png', names)
            self.assertIn('input/generation_mask.png', names)
            self.assertIn('input/prompt.txt', names)
            self.assertIn('input/context_board.png', names)
            self.assertIn('job_info.json', names)
            guide = z.read('CHATGPT_INSTRUCTIONS.md').decode()
            self.assertIn('54 x 42', guide)
            self.assertIn('Do not resize', guide)
            self.assertNotIn(str(self.jobs), z.read('job_info.json').decode())
        self.assertEqual(cli.pack_chatgpt(self.jobs, plan['job']), bundle)
        self.assertFalse((self.jobs / plan['job'] / 'region.json').exists())

    def test_expand_cli_creates_zip_and_dry_run_does_not(self):
        self.assertEqual(cli.main(['--jobs', str(self.jobs), 'expand',
                                  '--direction', 'east', '--provider', 'chatgpt',
                                  '--dry-run']), 0)
        self.assertFalse((self.jobs / 'tile_x+1_y+0').exists())
        self.assertEqual(cli.main(['--jobs', str(self.jobs), 'expand',
                                  '--direction', 'east', '--provider', 'chatgpt']), 0)
        bundle = self.jobs / 'tile_x+1_y+0' / 'chatgpt_bundle.zip'
        self.assertTrue(bundle.exists())
        self.assertEqual(cli.main(['--jobs', str(self.jobs), 'pack',
                                  '--job', 'tile_x+1_y+0']), 0)

    def test_reject_wrong_generated_size_without_resizing(self):
        plan = cli.prepare_expansion(self.jobs, 'east', context=12)
        path = self.jobs / plan['job'] / 'generated.png'
        Image.new('RGB', (self.w * 2, self.h * 2)).save(path)
        with self.assertRaisesRegex(ValueError, 'No resize or upscaling'):
            cli.publish(self.jobs, plan['job'])
        self.assertFalse((self.jobs / plan['job'] / 'region.json').exists())

    def test_portable_jobs_root_when_previous_absolute_path_is_invalid(self):
        plan = cli.prepare_expansion(self.jobs, 'east', context=12)
        meta_file = self.jobs / plan['job'] / 'grid_job.json'
        meta = load_json(meta_file)
        meta['jobs_root'] = '/some/other/machine/jobs'
        meta_file.write_text(json.dumps(meta))
        self.fake_output(plan)
        result = cli.publish(self.jobs, plan['job'], feather=0)
        self.assertTrue(Path(result['published']).is_file())

    def test_direction_from_newest_in_orthogonal_axis(self):
        first = cli.prepare_expansion(self.jobs, 'north', context=12)
        self.fake_output(first)
        cli.publish(self.jobs, first['job'], feather=0)
        second = cli.prepare_expansion(self.jobs, 'north', context=12)
        self.assertEqual(second['target'], {'x': 0, 'y': -2})


if __name__ == '__main__':
    unittest.main()
