"""World draft renderer tests: real overlap geometry and dependency order."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from draft_world_preview import create_world_preview
from tile_grid import save_json


class WorldPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.jobs = Path(self.temp.name) / 'jobs'
        self.jobs.mkdir()
        save_json(self.jobs / 'layout.json',
                  {'version': 1, 'width': 54, 'height': 42, 'overlap': 8})
        (self.jobs / 'center').mkdir()
        Image.new('RGB', (54, 42), (0, 0, 0)).save(
            self.jobs / 'center' / 'region.png')
        self.add('west_01', -1, 0, (10, 200, 10), ['center'])
        self.add('north_02', -1, -1, (200, 10, 10), ['west_01'])
        self.add('south_01', -1, 1, (10, 10, 200), ['west_01'])
        self.target = self.add('tile_x-2_y+0', -2, 0, (15, 15, 15),
                               ['west_01', 'north_02', 'south_01'])
        self.meta = {
            'target': {'x': -2, 'y': 0},
            'region_id': 'tile_x-2_y+0',
            'neighbors': {'east': {'id': 'west_01'}},
            'diagonal_neighbors': {
                'northeast': {'id': 'north_02'},
                'southeast': {'id': 'south_01'},
            },
        }
        draft = self.target / 'draft'
        draft.mkdir()
        self.draft = draft / 'tile.png'
        Image.new('RGB', (54, 42), (240, 130, 20)).save(self.draft)

    def add(self, name, x, y, rgb, deps):
        folder = self.jobs / name
        folder.mkdir()
        Image.new('RGB', (54, 42), rgb).save(folder / 'tile.png')
        save_json(folder / 'region.json', {
            'format': 'local_tile_v1', 'id': name,
            'x': x, 'y': y, 'image': 'tile.png',
            'dependencies': deps,
        })
        return folder

    def test_all_diagonal_neighbors_are_drawn_in_export_order(self):
        path = create_world_preview(self.target, self.draft, self.meta)
        with Image.open(path) as im:
            self.assertEqual(im.size, (146, 110))
            self.assertEqual(im.getpixel((50, 40)), (240, 130, 20))
            self.assertEqual(im.getpixel((102, 48)), (10, 200, 10))
            self.assertEqual(im.getpixel((102, 4)), (200, 10, 10))
            self.assertEqual(im.getpixel((102, 85)), (10, 10, 200))
        report = json.loads((self.target / 'draft/world_preview.json').read_text())
        self.assertEqual(report['dependencies'],
                         ['west_01', 'north_02', 'south_01'])
        self.assertEqual(report['painted'][-1], 'tile_x-2_y+0')

    def test_published_tile_is_not_modified_by_preview(self):
        old = (self.target / 'tile.png').read_bytes()
        create_world_preview(self.target, self.draft, self.meta)
        self.assertEqual((self.target / 'tile.png').read_bytes(), old)

    def test_dependency_cycle_is_reported_not_hidden(self):
        path = self.jobs / 'south_01/region.json'
        meta = json.loads(path.read_text())
        meta['dependencies'].append('tile_x-2_y+0')
        save_json(path, meta)
        with self.assertRaisesRegex(ValueError, 'paint order'):
            create_world_preview(self.target, self.draft, self.meta)
