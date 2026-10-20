"""The user receives the same pixel-coordinate instructions used in validation."""
import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from handoff_prompt import rectangles,render_prompt,refresh_prompt
from generation_check import check_job
from frontier_cli import main,publish
from prepare_region_grid import prepare
from tile_grid import save_json,load_json


class PromptGeometryTests(unittest.TestCase):
    def test_disjoint_rectangles_exactly_reconstruct_masks_with_holes(self):
        rng=np.random.default_rng(42)
        cases=[rng.random((18,23))>.6,np.zeros((4,5),bool),np.ones((4,5),bool)]
        hollow=np.ones((17,19),bool);hollow[3:12,5:14]=False;cases.append(hollow)
        for pixels in cases:
            counts=np.zeros(pixels.shape,int)
            for a,b,c,d in rectangles(pixels):counts[b:d,a:c]+=1
            np.testing.assert_array_equal(counts,pixels.astype(int))

    def test_prompt_edit_coordinates_match_attached_mask_not_tile_coordinates(self):
        mask=np.zeros((1024,1536),np.uint8);mask[42:983,:580]=255
        labels=np.zeros_like(mask);labels[:,580:]=1;labels[mask!=0]=3
        zones=dict(canvas_size=[1536,1024],pass_count=3,tile_edit_rect=[836,0,1416,941],
                   tile_canvas_rect=[836,-42,2372,982],canvas_world_origin=[-1996,-42])
        for current,left,right in [(1,0,580),(2,0,580),(3,0,256)]:
            zones['pass']=current
            mask[:]=0;mask[42:983,left:right]=255
            labels[:]=0;labels[:,right:]=1;labels[mask!=0]=3
            text=render_prompt({},zones,mask,labels)
            block=text.split('EDIT ONLY these WHITE-mask rectangles:\n')[1].split('Every pixel')[0]
            parsed=np.zeros(mask.shape,bool)
            for a,c,b,d in re.findall(r'x=(\d+) through (\d+), y=(\d+) through (\d+)',block):
                parsed[int(b):int(d)+1,int(a):int(c)+1]=True
            np.testing.assert_array_equal(parsed,mask==255)
            self.assertNotIn('[-1996, -42]',text)
            self.assertNotIn('[836, 0, 1416, 941]',text)
            self.assertNotIn('world origin',text.lower())
            self.assertIn('Do NOT enlarge',text)
            self.assertIn('Return the COMPLETE canvas at EXACTLY 1536x1024',text)


class SelfserveJobTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.jobs=Path(tmp.name)/'jobs';(self.jobs/'center').mkdir(parents=True)
        self.w,self.h,self.ov=54,42,8
        self.layout=dict(version=1,width=self.w,height=self.h,overlap=self.ov)
        save_json(self.jobs/'layout.json',self.layout)
        Image.fromarray(self.world(0,0,self.w,self.h)).save(self.jobs/'center/region.png')
        self.folder=self.jobs/'new'
        prepare(self.jobs,self.folder,-1,0,'east')
        meta=load_json(self.folder/'grid_job.json')
        ox,oy=meta['generation_world_origin'];cw,ch=meta['expected_generation_size']
        result=self.world(ox,oy,cw,ch)
        mask=np.array(Image.open(self.folder/'input/generation_mask.png'))
        canvas=np.array(Image.open(self.folder/'input/generation_canvas.png'))
        result[mask==0]=canvas[mask==0]
        Image.fromarray(result).save(self.folder/'generated.png')

    @staticmethod
    def world(x,y,w,h):
        xx,yy=np.meshgrid(np.arange(w)+x,np.arange(h)+y)
        return np.stack([(xx+yy*3)%256,(xx*2+yy)%256,(xx+yy)%256],axis=2).astype('uint8')

    def immutable(self):
        names=['grid_job.json','generated.png','input/generation_canvas.png',
               'input/generation_mask.png','input/generation_zones.png','input/zones.json',
               'input/accepted_tile.png','input/remaining_mask.png','input/patch_state.json']
        names += [n for n in ('tile.png','region.json') if (self.folder/n).exists()]
        return {n:(self.folder/n).read_bytes() for n in names}

    def assert_immutable(self,before):
        for name,data in before.items():self.assertEqual((self.folder/name).read_bytes(),data,name)

    def test_refresh_only_text_preserves_published_terrain_and_progress(self):
        publish(self.jobs,'new',export=False)
        before=self.immutable()
        (self.folder/'input/prompt.txt').write_text('old conflicting coordinates')
        result=refresh_prompt(self.folder)
        self.assertTrue(result['changed'])
        self.assertEqual(Path(result['backup']).read_text(),'old conflicting coordinates')
        self.assert_immutable(before)
        self.assertFalse(refresh_prompt(self.folder)['changed'])
        self.assert_immutable(before)

    def test_failed_refresh_restores_original_text_without_image_changes(self):
        before=self.immutable()
        path=self.folder/'input/prompt.txt';path.write_text('old prompt')
        replace=Path.replace
        def fail(src,dst):
            if src.name=='prompt.txt.tmp':raise OSError('injected prompt failure')
            return replace(src,dst)
        with patch.object(Path,'replace',fail):
            with self.assertRaisesRegex(OSError,'injected'):refresh_prompt(self.folder)
        self.assertEqual(path.read_text(),'old prompt')
        self.assertFalse((self.folder/'input/prompt.txt.tmp').exists())
        self.assert_immutable(before)

    def test_check_valid_result_has_next_step_and_does_not_advance(self):
        before=self.immutable()
        result=check_job(self.jobs,'new',strict=True)
        self.assertEqual(result['status'],'passed')
        self.assertTrue(result['tile_complete'])
        self.assertIsNone(result['retry_prompt'])
        self.assert_immutable(before)
        self.assertFalse((self.folder/'region.json').exists())

    def test_wrong_dimensions_write_retry_and_cli_returns_nonzero(self):
        Image.new('RGB',(1562,1007)).save(self.folder/'generated.png')
        before=self.immutable()
        with contextlib.redirect_stdout(io.StringIO()):
            code=main(['--jobs',str(self.jobs),'check-job','--job','new','--strict'])
        self.assertEqual(code,1)
        result=load_json(self.folder/'draft/generation_check.json')
        self.assertEqual(result['issues'],['wrong_dimensions'])
        retry=(self.folder/'draft/retry_prompt.txt').read_text()
        self.assertTrue(retry.startswith((self.folder/'input/prompt.txt').read_text()))
        self.assertIn('1536x1024',retry)
        self.assert_immutable(before)

    def test_rejected_anchor_records_reason_and_preserves_state(self):
        result=np.array(Image.open(self.folder/'generated.png'))
        zones=np.array(Image.open(self.folder/'input/generation_zones.png'))
        result[zones==1]=255-result[zones==1]
        Image.fromarray(result).save(self.folder/'generated.png')
        (self.folder/'input/prompt.txt').write_text('old conflicting world origin [-1996,-42]')
        before=self.immutable()
        r=check_job(self.jobs,'new',strict=True)
        self.assertEqual(r['status'],'rejected')
        self.assertIn('weak_alignment',r['issues'])
        self.assertTrue(r['alignment']['failed_checks'])
        self.assertEqual(r['alignment']['limits']['maximum_fitted_drift_px'],2.)
        self.assertTrue((self.folder/'draft/locked_context_difference.png').is_file())
        self.assertIn('generation_canvas.png',Path(r['retry_prompt']).read_text())
        self.assertNotIn('old conflicting world origin',Path(r['retry_prompt']).read_text())
        self.assertIn('ALL COORDINATES BELOW REFER ONLY',Path(r['retry_prompt']).read_text())
        self.assertEqual((self.folder/'input/prompt.txt').read_text(),'old conflicting world origin [-1996,-42]')
        self.assert_immutable(before)
        # A valid new attempt removes the obsolete failure retry text.
        fixed=np.array(Image.open(self.folder/'input/generation_canvas.png'))
        result[zones==1]=fixed[zones==1]
        Image.fromarray(result).save(self.folder/'generated.png')
        self.assertEqual(check_job(self.jobs,'new',strict=True)['status'],'passed')
        self.assertFalse((self.folder/'draft/retry_prompt.txt').exists())
