"""Native passes cover new world pixels once, preserving all existing geography."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from generation_patches import plan_passes, apply_patch_result
from frontier_cli import prepare_expansion, advance_job, publish, rebuild_job
from tile_grid import load_json,save_json,DIRS
from prepare_region_grid import prepare
from native_identity import verify_identity


class NativePassTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.jobs=Path(temp.name)/'jobs';(self.jobs/'center').mkdir(parents=True)
        self.w,self.h,self.ov=1672,941,256
        self.layout=dict(version=1,width=self.w,height=self.h,overlap=self.ov)
        save_json(self.jobs/'layout.json',self.layout)
        Image.fromarray(self.world(0,0,self.w,self.h)).save(self.jobs/'center/region.png')

    @staticmethod
    def world(x,y,w,h):
        xx,yy=np.meshgrid(np.arange(w)+x,np.arange(h)+y)
        return np.stack([(xx+yy*3)%256,(xx*2+yy)%256,(xx//7+yy//11)%256],axis=2).astype('uint8')

    def fake(self,folder):
        meta=load_json(folder/'grid_job.json');ox,oy=meta['generation_world_origin']
        cw,ch=meta['expected_generation_size'];im=self.world(ox,oy,cw,ch)
        mask=np.array(Image.open(folder/'input/generation_mask.png'))
        canvas=np.array(Image.open(folder/'input/generation_canvas.png'))
        im[mask==0]=canvas[mask==0]
        Image.fromarray(im).save(folder/'generated.png')
        return im

    def complete(self,folder):
        while True:
            generated=self.fake(folder)
            _,remaining,_=apply_patch_result(folder,generated)
            if not remaining.any():break
            advance_job(self.jobs,folder.name)
        return generated

    def test_plan_native_dimensions_context_and_coverage_all_sides(self):
        for side in DIRS:
            with self.subTest(side=side):
                plans=plan_passes(self.layout,side)
                coverage=np.zeros((self.h,self.w),bool)
                for p in plans:
                    a,b,c,d=p['tile_canvas_rect'];l,t,r,s=p['tile_edit_rect']
                    self.assertEqual((c-a,d-b),(1536,1024) if side in ('east','west') else (1024,1536))
                    self.assertTrue(a<=l<r<=c and b<=t<s<=d)
                    coverage[t:s,l:r]=True
                expected=np.ones_like(coverage)
                if side=='east':expected[:,-256:]=False
                elif side=='west':expected[:,:256]=False
                elif side=='north':expected[:256]=False
                else:expected[-256:]=False
                np.testing.assert_array_equal(coverage,expected)
        self.assertEqual(len(plan_passes(self.layout,'east')),3)

    def test_four_directions_exact_native_roundtrip(self):
        for direction in DIRS:
            with self.subTest(direction=direction):
                plan=prepare_expansion(self.jobs,direction,anchor='center')
                folder=self.jobs/plan['job'];self.complete(folder)
                publish(self.jobs,folder.name,draft=True,export=False)
                image=np.array(Image.open(folder/'draft/tile.png'))
                dx,dy=DIRS[direction]
                expected=self.world(dx*(self.w-self.ov),dy*(self.h-self.ov),self.w,self.h)
                np.testing.assert_array_equal(image,expected)
                self.assertFalse((folder/'region.json').exists())

    def source(self,name,x,y,deps=(),rgb=None):
        folder=self.jobs/name;folder.mkdir()
        data=self.world(x*(self.w-self.ov),y*(self.h-self.ov),self.w,self.h)
        if rgb is not None:data[:]=rgb
        Image.fromarray(data).save(folder/'tile.png')
        save_json(folder/'region.json',dict(format='local_tile_v1',id=name,x=x,y=y,image='tile.png',dependencies=list(deps)))
        return folder

    def test_published_west_rebuild_real_diagonals_and_safe_advance(self):
        self.source('west_01',-1,0,['center'])
        self.source('north_02',-1,-1,['west_01'])
        self.source('south_01',-1,1,['west_01'])
        folder=self.source('tile_x-2_y+0',-2,0,['west_01','north_02','south_01'],213)
        save_json(folder/'grid_job.json',dict(region_id=folder.name,target=dict(x=-2,y=0),primary_side='east'))
        (folder/'input').mkdir();(folder/'input/old.txt').write_text('old')
        before={n:(folder/n).read_bytes() for n in ('tile.png','region.json')}
        rebuild_job(self.jobs,folder.name)
        zones=load_json(folder/'input/zones.json')
        self.assertEqual(set(zones['neighbors']),{'east','northeast','southeast'})
        self.assertEqual(next(s['source_crop'] for s in zones['sources'] if s['id']=='west_01'),[0,0,956,941])
        self.assertNotIn(folder.name,[s['id'] for s in zones['sources']])
        self.fake(folder)
        with self.assertRaisesRegex(ValueError,'incomplete'):publish(self.jobs,folder.name,replace=True,export=False)
        advance_job(self.jobs,folder.name)
        self.assertFalse((folder/'generated.png').exists())
        self.complete(folder)
        publish(self.jobs,folder.name,draft=True,export=False,strict=True)
        for n,data in before.items():self.assertEqual((folder/n).read_bytes(),data)
        lock=np.array(Image.open(folder/'input/neighbor_lock.png'));mask=np.array(Image.open(folder/'input/neighbor_mask.png'))
        tile=np.array(Image.open(folder/'draft/tile.png'))
        np.testing.assert_array_equal(tile[mask==0],lock[mask==0])
        np.testing.assert_array_equal(tile,self.world(-2832,0,self.w,self.h))

    def test_advance_rolls_back_partial_install(self):
        folder=self.jobs/'new';prepare(self.jobs,folder,-1,0,'east');self.fake(folder)
        original_meta=(folder/'grid_job.json').read_bytes()
        original_input=(folder/'input/generation_canvas.png').read_bytes()
        original_generation=(folder/'generated.png').read_bytes()
        rename=Path.rename
        def fail(src,dest):
            if src.parent.name.startswith('advance_stage_') and src.name=='grid_job.json':raise OSError('injected failure')
            return rename(src,dest)
        with patch.object(Path,'rename',fail):
            with self.assertRaisesRegex(OSError,'injected'):advance_job(self.jobs,'new')
        self.assertEqual((folder/'grid_job.json').read_bytes(),original_meta)
        self.assertEqual((folder/'input/generation_canvas.png').read_bytes(),original_input)
        self.assertEqual((folder/'generated.png').read_bytes(),original_generation)
        self.assertFalse(list(self.jobs.glob('advance_stage_*')))

    def test_reject_wrong_size_changed_anchor_and_modified_mask(self):
        folder=self.jobs/'new';prepare(self.jobs,folder,-1,0,'east')
        generated=self.fake(folder);old=(folder/'grid_job.json').read_bytes()
        Image.new('RGB',(1562,1007)).save(folder/'generated.png')
        with self.assertRaisesRegex(ValueError,'No resize'):advance_job(self.jobs,'new')
        zones=np.array(Image.open(folder/'input/generation_zones.png'))
        generated[zones==1]=255-generated[zones==1]
        Image.fromarray(generated).save(folder/'generated.png')
        with self.assertRaisesRegex(ValueError,'Locked world context changed'):advance_job(self.jobs,'new')
        self.fake(folder)
        mask=np.array(Image.open(folder/'input/generation_mask.png'));mask[mask==255]=0
        Image.fromarray(mask).save(folder/'input/generation_mask.png')
        with self.assertRaisesRegex(ValueError,'mask differs'):advance_job(self.jobs,'new')
        self.assertEqual((folder/'grid_job.json').read_bytes(),old)

    def test_eight_neighbors_corner_ownership_across_passes(self):
        tx,ty=-2,0
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx or dy:self.source(f'n{dx}_{dy}',tx+dx,ty+dy)
        folder=self.jobs/'new';prepare(self.jobs,folder,tx,ty,'east')
        self.complete(folder)
        publish(self.jobs,'new',draft=True,export=False)
        tile=np.array(Image.open(folder/'draft/tile.png'))
        np.testing.assert_array_equal(tile,self.world(tx*(self.w-self.ov),ty*(self.h-self.ov),self.w,self.h))


class NativeIdentityTests(unittest.TestCase):
    def test_photometric_change_accepts_identity_but_not_shift_or_scale(self):
        import cv2
        rng=np.random.default_rng(3)
        reference=rng.integers(0,210,(500,700,3),dtype=np.uint8)
        reference=cv2.GaussianBlur(reference,(3,3),0)
        mask=np.ones(reference.shape[:2],bool)
        toned=np.clip(reference.astype('int16')+22,0,255).astype('uint8')
        report=verify_identity(toned,reference,mask)
        self.assertEqual(report['status'],'accepted')
        self.assertFalse(report['registration_applied'])
        shifted=cv2.warpAffine(toned,np.float32([[1,0,7],[0,1,0]]),(700,500))
        self.assertEqual(verify_identity(shifted,reference,mask)['status'],'rejected')
        scaled=cv2.warpAffine(toned,np.float32([[1.004,0,0],[0,1.004,0]]),(700,500))
        self.assertEqual(verify_identity(scaled,reference,mask)['status'],'rejected')
