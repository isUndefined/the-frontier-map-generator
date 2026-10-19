import unittest
import cv2
import numpy as np
from terrain_alignment import align_terrain, AlignmentError


class TerrainAlignmentTests(unittest.TestCase):
    def texture(self, seed=73):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        for _ in range(300):
            x, y = rng.integers([10, 10], [630, 470])
            color = tuple(int(v) for v in rng.integers(0, 256, 3))
            cv2.circle(image, (int(x), int(y)), int(rng.integers(3, 10)), color, -1)
        return image

    def test_known_scale_and_translation_recovered_in_all_directions(self):
        reference = self.texture()
        matrix = np.array([[.992, .002, 1.5], [-.001, .987, 1.5]], np.float64)
        generated = cv2.warpAffine(reference, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT_101)
        for side in ('east', 'west', 'north', 'south'):
            with self.subTest(side=side):
                fixed = np.zeros((480, 640), bool)
                if side == 'east': fixed[:, 320:] = True; core, overlap = 320, 128
                elif side == 'west': fixed[:, :320] = True; core, overlap = 320, 128
                elif side == 'north': fixed[:160] = True; core, overlap = 320, 160
                else: fixed[320:] = True; core, overlap = 320, 160
                error = float(np.abs(generated.astype(float) - reference)[fixed].mean())
                aligned, report = align_terrain(generated, reference, fixed, side, core, overlap, error)
                estimated = np.array(report['reference_to_generated'])
                corners = np.array([[100, 100, 1], [400, 250, 1], [580, 380, 1]])
                self.assertLess(np.linalg.norm(corners @ (estimated - matrix).T, axis=1).max(), 1)
                self.assertLess(report['median_landmark_error_px'], 1)
                self.assertLess(report['anchor_mae_after'], error * .5)
                self.assertEqual(aligned.shape, reference.shape)

    def test_unrelated_terrain_rejected(self):
        reference, generated = self.texture(), self.texture(918)
        fixed = np.zeros(reference.shape[:2], bool)
        fixed[:, 320:] = True
        error = float(np.abs(generated.astype(float) - reference)[fixed].mean())
        with self.assertRaises(AlignmentError):
            align_terrain(generated, reference, fixed, 'east', 320, 128, error)

    def test_flat_anchor_rejected_instead_of_inventing_transform(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        fixed = np.zeros(reference.shape[:2], bool)
        fixed[:, 320:] = True
        with self.assertRaises(AlignmentError):
            align_terrain(self.texture(), reference, fixed, 'east', 320, 128, 80)


if __name__ == '__main__':
    unittest.main()
