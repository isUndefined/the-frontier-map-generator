import unittest
import numpy as np
from seam_refinement import tone_match_generated
from seam_diagnostics import analyze_seam

class SeamDiagnosticsTests(unittest.TestCase):
    def test_bounded_tone_match(self):
        generated=np.full((130,160,3), 100, np.uint8)
        locked=np.full_like(generated, 120)
        changed, info=tone_match_generated(generated,locked,128,152,40,90)
        self.assertTrue(info['enabled'])
        self.assertTrue(np.array_equal(changed[:, :32], generated[:, :32]))
        self.assertGreater(int(changed[65,130,0]),100)
        self.assertLessEqual(float(max(info['offset_rgb'])),18)

    def test_severe_straight_seam_is_rejected(self):
        data=np.full((130,160,3),20,np.uint8)
        data[:,128:]=240
        report=analyze_seam(data,'east',32)
        self.assertEqual(report['status'],'rejected')
        self.assertGreater(report['discontinuity_fraction'],.8)

    def test_identical_continuous_terrain_passes(self):
        data=np.zeros((130,160,3),np.uint8)
        data[:]=np.arange(160,dtype=np.uint8)[None,:,None]
        self.assertEqual(analyze_seam(data,'east',32)['status'],'passed')

if __name__ == '__main__':
    unittest.main()
