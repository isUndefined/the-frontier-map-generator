import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from neighbor_canvas import compose_lock
from frontier_cli import rebuild_job
from tile_grid import save_json

class NeighborCanvasTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.jobs = Path(t.name) / 'jobs'
        (self.jobs / 'center').mkdir(parents=True)
        self.w, self.h, self.ov = 54, 42, 8
        save_json(self.jobs / 'layout.json', dict(version=1,width=self.w,height=self.h,overlap=self.ov))
        Image.new('RGB',(self.w,self.h),'gray').save(self.jobs/'center/region.png')
        self.make('west_01',-1,0,(40,120,60),['center'])
        self.make('north_02',-1,-1,(195,40,100),['west_01'])
        self.make('south_01',-1,1,(10,100,190),['west_01'])
        self.target=self.make('tile_x-2_y+0',-2,0,(115,70,30),['west_01','north_02','south_01'])
        save_json(self.target/'grid_job.json',{
            'method':'local_tile_v1','region_id':'tile_x-2_y+0',
            'target':{'x':-2,'y':0},'layout':dict(version=1,width=self.w,height=self.h,overlap=self.ov),
            'primary_side':'east','context':12})
        (self.target/'input').mkdir()
        (self.target/'input'/'old.txt').write_text('old input')
        Image.new('RGB',(self.w,self.h),'red').save(self.target/'generated.png')

    def make(self,name,x,y,rgb,deps):
        p=self.jobs/name
        p.mkdir(exist_ok=True)
        Image.new('RGB',(self.w,self.h),rgb).save(p/'tile.png')
        save_json(p/'region.json',{'format':'local_tile_v1','id':name,'x':x,'y':y,
                   'image':'tile.png','dependencies':deps})
        return p

    def test_composite_is_real_world_not_old_target(self):
        lock, mask, owners, intersections=compose_lock(self.jobs,-2,0)
        self.assertEqual(set(i['id'] for i in intersections), {'west_01','north_02','south_01'})
        np.testing.assert_array_equal(lock[0,-1],(195,40,100))
        np.testing.assert_array_equal(lock[self.h//2,-1],(40,120,60))
        np.testing.assert_array_equal(lock[-1,-1],(10,100,190))
        self.assertEqual(mask[0,-1],0)
        self.assertEqual(mask[self.h//2,0],255)

    def test_rebuild_keeps_published_and_backs_up_input(self):
        oldtile=(self.target/'tile.png').read_bytes()
        oldregion=(self.target/'region.json').read_bytes()
        r=rebuild_job(self.jobs,'tile_x-2_y+0')
        self.assertEqual((self.target/'tile.png').read_bytes(),oldtile)
        self.assertEqual((self.target/'region.json').read_bytes(),oldregion)
        back=Path(r['backup'])
        self.assertEqual((back/'input/old.txt').read_text(),'old input')
        self.assertTrue((back/'generated.png').exists())
        self.assertFalse((self.target/'generated.png').exists())
        mask=np.asarray(Image.open(r['mask']))
        canvas=np.asarray(Image.open(r['canvas']))
        np.testing.assert_array_equal(canvas[0,-1],(195,40,100))
        np.testing.assert_array_equal(canvas[-1,-1],(10,100,190))
        self.assertEqual(mask[0,-1],0)
        self.assertTrue(Path(r['prompt']).exists())
        self.assertTrue(Path(r['debug']).exists())
        self.assertEqual(json.loads((self.target/'grid_job.json').read_text())['neighbor_lock_version'],1)
