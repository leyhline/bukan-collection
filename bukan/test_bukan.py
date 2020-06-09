import unittest
from pathlib import Path
import numpy as np
import cv2 as cv
from bukan import detect_features, match_pair


SCRIPT_PATH = Path(__file__).parent
IMAGE_PATH1 = "200019642_00022.jpg"
IMAGE_PATH2 = "200019648_00022.jpg"
"""These are two JPG encoded images from the Bukan Collection used as test data."""


def read_jpg_image(path: Path) -> np.ndarray:
    with open(path, "rb") as fd:
        jpg1 = np.frombuffer(fd.read(), np.uint8)
        return cv.imdecode(jpg1, cv.IMREAD_GRAYSCALE)


class TestBukan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img1 = read_jpg_image(SCRIPT_PATH.joinpath(IMAGE_PATH1))
        cls.img2 = read_jpg_image(SCRIPT_PATH.joinpath(IMAGE_PATH2))

    def test_detect_features_img1(self):
        keypoints1, descriptors1 = detect_features(self.img1)
        self.assertEqual(len(keypoints1), descriptors1.shape[0])
        self.assertEqual(descriptors1.shape[1], 61)
        self.assertEqual(len(keypoints1), 333)
        self.assertIsInstance(keypoints1[0], cv.KeyPoint)

    def test_detect_features_img2(self):
        keypoints2, descriptors2 = detect_features(self.img2)
        self.assertEqual(len(keypoints2), descriptors2.shape[0])
        self.assertEqual(descriptors2.shape[1], 61)
        self.assertEqual(len(keypoints2), 121)
        self.assertIsInstance(keypoints2[0], cv.KeyPoint)

    def test_detect_features_empty_image(self):
        img = np.zeros((200, 400), dtype=np.uint8)
        self.assertRaises(Exception, detect_features, img)

    def test_match_pair(self):
        keypoints1, descriptors1 = detect_features(self.img1)
        keypoints2, descriptors2 = detect_features(self.img2)
        matches, h = match_pair(keypoints1, descriptors1, keypoints2, descriptors2)
        self.assertEqual(len(matches), 48)
        self.assertIsInstance(matches[0], cv.DMatch)
        self.assertFalse(np.any(np.isnan(h)))

    def test_match_pair_against_empty(self):
        empty_keypoints = []
        empty_descriptors = np.empty((0, 61), dtype=np.uint8)
        keypoints2, descriptors2 = detect_features(self.img2)
        self.assertRaises(Exception, match_pair, empty_keypoints, empty_descriptors, keypoints2, descriptors2)


if __name__ == "__main__":
    unittest.main()
