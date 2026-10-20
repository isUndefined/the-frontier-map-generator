import unittest

import numpy as np

from seam_transition import editable_overlap_mask, merge_editable_overlap, _orient, _unorient
from seam_refinement import tone_match_generated


class OverlapSeamTests(unittest.TestCase):
    def test_exact_outer_border_and_diagonal_corners_for_all_sides(self):
        height, width, ov = 130, 160, 32
        for side in ('east', 'west', 'north', 'south'):
            with self.subTest(side=side):
                base = np.full((height, width), 255, np.uint8)
                oriented = _orient(base, side)
                oriented[:, -ov:] = 0
                lock_mask = np.ascontiguousarray(
                    {'east': lambda v: v,
                     'west': lambda v: np.flip(v, axis=1),
                     'north': lambda v: np.rot90(v, 1),
                     'south': lambda v: np.rot90(v, 3)}[side](oriented)
                )
                generation_mask, config = editable_overlap_mask(lock_mask, side, ov)
                self.assertIsNotNone(config)
                self.assertGreater(int(np.sum((generation_mask == 255) & (lock_mask == 0))), 0)
                locked = np.zeros((height, width, 3), np.uint8)
                locked[:] = (20, 90, 120)
                generated = np.full_like(locked, 190)
                result, report = merge_editable_overlap(
                    generated, locked, lock_mask, generation_mask, side)
                # The black/preserve pixels are restored exactly everywhere,
                # including every diagonal 256x256 corner equivalent.
                np.testing.assert_array_equal(result[generation_mask == 0],
                                              locked[generation_mask == 0])
                self.assertEqual(report['method'], 'terrain_aware_hard_cut')
                self.assertEqual(report['seam_x_min'], min(np.where(
                    _orient((lock_mask == 0) & (generation_mask == 255), side))[1]))
                # The existing workflow includes a documented 96px generated-only
                # tone ramp. Compare against that ramp, not the pre-ramp image.
                ys,xs=np.nonzero(_orient((lock_mask==0)&(generation_mask==255),side))
                toned,_=tone_match_generated(_orient(generated,side).copy(),
                    _orient(locked,side),int(xs.min()),int(xs.max())+1,
                    int(ys.min()),int(ys.max())+1,radius=96)
                expected=_unorient(toned,side)
                np.testing.assert_array_equal(result[lock_mask==255],expected[lock_mask==255])

    def test_path_can_move_into_overlap_instead_of_fixed_vertical_seam(self):
        h, w, ov = 130, 160, 32
        mask = np.full((h, w), 255, np.uint8)
        mask[:, w-ov:] = 0
        editable, _ = editable_overlap_mask(mask, 'east', ov)
        generated = np.full((h, w, 3), 180, np.uint8)
        locked = np.full((h, w, 3), 30, np.uint8)
        locked[:, w-ov+10] = 180
        result, report = merge_editable_overlap(generated, locked, mask, editable, 'east')
        self.assertGreater(report['seam_x_max'], w-ov)
        np.testing.assert_array_equal(result[:, -8:], locked[:, -8:])
        ys,xs=np.nonzero((mask==0)&(editable==255))
        toned,_=tone_match_generated(generated.copy(),locked,int(xs.min()),int(xs.max())+1,
                                    int(ys.min()),int(ys.max())+1,radius=96)
        np.testing.assert_array_equal(result[:, :w-ov],toned[:, :w-ov])
        np.testing.assert_array_equal(result[:, :w-ov-96],generated[:, :w-ov-96])

    def test_no_editable_corridor_preserves_original_lock(self):
        mask = np.full((42, 54), 255, np.uint8)
        mask[:, -8:] = 0
        editable, config = editable_overlap_mask(mask, 'east', 8)
        self.assertIsNone(config)
        np.testing.assert_array_equal(editable, mask)
        result, report = merge_editable_overlap(
            np.full((42,54,3), 220, np.uint8),
            np.full((42,54,3), 70, np.uint8), mask, editable, 'east')
        self.assertEqual(report['method'], 'exact_lock')
        self.assertTrue((result[:,-8:] == 70).all())

    def test_rejects_attempt_to_lock_unknown_pixels(self):
        mask = np.full((130,160), 255, np.uint8)
        mask[:, -32:] = 0
        generated_mask = mask.copy()
        generated_mask[20,20] = 0
        with self.assertRaisesRegex(ValueError, 'unknown pixels'):
            merge_editable_overlap(
                np.zeros((130,160,3),np.uint8), np.zeros((130,160,3),np.uint8),
                mask, generated_mask, 'east')


if __name__ == '__main__':
    unittest.main()
