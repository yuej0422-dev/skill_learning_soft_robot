from __future__ import annotations

import unittest

import numpy as np

from soft_vla.data.image_preprocessor import ensure_rgb_uint8, images_are_distinct


class ImagePreprocessorTest(unittest.TestCase):
    def test_rgb_uint8(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(ensure_rgb_uint8(img).shape, (8, 8, 3))

    def test_distinct(self):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8)
        self.assertTrue(images_are_distinct([a, b]))


if __name__ == "__main__":
    unittest.main()

