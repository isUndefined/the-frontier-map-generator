"""World-coordinate contract, native masks, all sides/corners and rebuild rollback."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from prepare_region_grid import prepare as _prepare
from functools import partial
prepare = partial(_prepare, prepare_version=2)
from prepare_v2 import build_generation_zones
from stitch_region_grid import stitch
from frontier_cli import rebuild_job, publish
from neighbor_canvas import compose_lock
from tile_grid import save_json, load_json, DIRS


class PrepareV2Tests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.jobs = Path(t.name) / 'jobs'
        (self.jobs / 'center').mkdir(parents=True)
        self.w, self.h, self.ov = 1000, 941, 256
        self.layout = dict(version=1,width=self.w,height=self.h,overlap=self.ov)
        save_json(self.jobs / 'layout.json', self.layout)
        Image.fromarray(self.world(0,0,self.w,self.h)).save(self.jobs/'center/region.png')

    @staticmethod
    def world(x,y,w,h):
        xx,yy = np.meshgrid(np.arange(w)+x,np.arange(h)+y)
        return np.stack([(xx+3*yy)%256,(2*xx+yy)%256,(xx//17+yy//13)%256],axis=2).astype(np.uint8)

    def source(self,name,x,y,color=None,deps=None):
        folder=self.jobs/name
        folder.mkdir()
        im=(self.world(x*(self.w-self.ov),y*(self.h-self.ov),self.w,self.h)
            if color is None else np.full((self.h,self.w,3),color,np.uint8))
        Image.fromarray(im).save(folder/'tile.png')
        save_json(folder/'region.json',dict(format='local_tile_v1',id=name,x=x,y=y,
                    image='tile.png',dependencies=deps or []))
        return folder

    def generated(self,folder):
        meta=load_json(folder/'grid_job.json')
        ox,oy=meta['generation_world_origin']
        cw,ch=meta['expected_generation_size']
        result=self.world(ox,oy,cw,ch)
        mask=np.asarray(Image.open(folder/'input/generation_mask.png'))
        canvas=np.asarray(Image.open(folder/'input/generation_canvas.png'))
        result[mask==0]=canvas[mask==0]
        Image.fromarray(result).save(folder/'generated.png')
        return result

    def test_all_four_directions_exact_world_crop_roundtrip(self):
        for direction,(dx,dy) in DIRS.items():
            side={'north':'south','south':'north','west':'east','east':'west'}[direction]
            with self.subTest(direction=direction):
                folder=self.jobs/('new_'+direction)
                prepare(self.jobs,folder,dx,dy,side)
                result=self.generated(folder)
                zones=load_json(folder/'input/zones.json')
                self.assertEqual(zones['canvas_size'],[self.w+384,self.h+384])
                self.assertEqual(zones['tile_crop'],[192,192,192+self.w,192+self.h])
                lock,mask,_,_=compose_lock(self.jobs,dx,dy)
                a,b,c,d=zones['tile_crop']
                cmask=np.asarray(Image.open(folder/'input/generation_mask.png'))
                np.testing.assert_array_equal(cmask[b:d,a:c],mask)
                np.testing.assert_array_equal(result[b:d,a:c][mask==0],lock[mask==0])
                labels=np.asarray(Image.open(folder/'input/generation_zones.png'))
                self.assertEqual(int(np.sum(labels==2)),192*(self.h if dx else self.w))
                self.assertTrue((labels==3).any())
                stitch(folder,draft=True,jobs=self.jobs,feather=0)
                tile=np.asarray(Image.open(folder/'draft/tile.png'))
                expected=self.world(dx*(self.w-self.ov),dy*(self.h-self.ov),self.w,self.h)
                np.testing.assert_array_equal(tile,expected)
                report=load_json(folder/'draft/stitch_meta.json')
                self.assertEqual(report['alignment']['status'],'accepted')

    def test_eight_neighbors_corners_and_exact_source_crops(self):
        x,y=-2,0
        for name,(dx,dy) in dict(north=(0,-1),south=(0,1),east=(1,0),west=(-1,0),
                    northwest=(-1,-1),northeast=(1,-1),southwest=(-1,1),southeast=(1,1)).items():
            self.source(name,x+dx,y+dy)
        canvas,mask,labels,zones=build_generation_zones(self.jobs,x,y,self.layout)
        self.assertEqual(len(zones['neighbors']),8)
        ox,oy=zones['canvas_world_origin']
        expected=self.world(ox,oy,*zones['canvas_size'])
        np.testing.assert_array_equal(canvas[labels==1],expected[labels==1])
        for item in zones['sources']:
            sx,sy=item['grid'];a,b,c,d=item['source_crop'];u,v,r,s=item['canvas_rect']
            self.assertEqual((r-u,s-v),(c-a,d-b))
            self.assertEqual((ox+u,oy+v),(sx*(self.w-self.ov)+a,sy*(self.h-self.ov)+b))
            if item['id'] in ('northwest','northeast','southwest','southeast'):
                a,b,c,d=item['overlap_source_crop']
                self.assertEqual((c-a,d-b),(256,256))
        self.assertTrue((mask[labels==1]==0).all())
        self.assertFalse(np.any((labels==2)&(mask==0)))

    def test_diagonal_only_transition_and_locked_padding(self):
        self.source('diagonal',-1,-1)
        canvas,mask,labels,zones=build_generation_zones(self.jobs,-2,0,self.layout)
        # NE diagonal only: transition extends left and down from its actual corner.
        a,b,c,d=zones['tile_crop']
        self.assertEqual(labels[b+256,c-256-192],2)
        self.assertEqual(labels[b+256,c-256-193],3)
        self.assertEqual(labels[b+256+192,c-1],3)
        self.assertTrue((mask[:b]==0).all())
        self.assertTrue((mask[d:]==0).all())
        self.assertTrue((mask[:, :a]==0).all())

    def test_rebuild_rollback_after_partial_install(self):
        folder=self.source('old',-1,0,deps=['center'])
        save_json(folder/'grid_job.json',dict(region_id='old',target=dict(x=-1,y=0),primary_side='east'))
        (folder/'input').mkdir();(folder/'input/old.txt').write_text('original')
        originals={p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        rename=Path.rename
        def fail(path,target):
            if path.parent.name.startswith('rebuild_stage_') and path.name=='grid_job.json':
                raise OSError('injected install failure')
            return rename(path,target)
        with patch.object(Path,'rename',fail):
            with self.assertRaisesRegex(OSError,'injected'): rebuild_job(self.jobs,'old')
        for n,data in originals.items():self.assertEqual((folder/n).read_bytes(),data)
        self.assertEqual((folder/'input/old.txt').read_text(),'original')
        self.assertFalse(list(self.jobs.glob('rebuild_stage_*')))

    def test_changed_world_anchor_is_not_warped_or_published(self):
        folder=self.jobs/'new'
        prepare(self.jobs,folder,-1,0,'east')
        result=self.generated(folder)
        labels=np.asarray(Image.open(folder/'input/generation_zones.png'))
        result[labels==1]=255-result[labels==1]
        Image.fromarray(result).save(folder/'generated.png')
        with self.assertRaisesRegex(ValueError,'Locked world context changed'):stitch(folder,jobs=self.jobs)
        self.assertFalse((folder/'tile.png').exists())
        with self.assertRaisesRegex(ValueError,'weak_alignment'):publish(self.jobs,'new',export=False)
        self.assertFalse((folder/'region.json').exists())
        report=load_json(folder/'draft/stitch_meta.json')
        self.assertEqual(report['alignment']['method'],'world_canvas_identity')
