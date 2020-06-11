import unittest
from base64 import b64decode
from test_bukan import read_jpg_image, IMAGE_PATH1, IMAGE_PATH2, SCRIPT_PATH
from bukan import detect_features
from keypoint import zip_features


class TestKeypoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img1 = read_jpg_image(SCRIPT_PATH.joinpath(IMAGE_PATH1))
        
    def test_zip_features(self):
        keypoints, descriptors = detect_features(self.img1)
        features = zip_features(keypoints, descriptors)
        self.assertEqual(333, len(features))
        kp1 = keypoints[5]
        desc1 = descriptors[5, :]
        feature1 = features[5]
        self.assertAlmostEqual(feature1["x"], kp1.pt[0])
        self.assertAlmostEqual(feature1["y"], kp1.pt[1])
        self.assertAlmostEqual(feature1["size"], kp1.size)
        self.assertAlmostEqual(feature1["angle"], kp1.angle)
        self.assertAlmostEqual(feature1["response"], kp1.response)
        self.assertEqual(feature1["octave"], kp1.octave)
        self.assertEqual(feature1["class_id"], kp1.class_id)
        feature1_desc = b64decode(feature1["base64descriptor"])
        self.assertEqual(61, len(feature1_desc))
        self.assertEqual(feature1_desc, bytes(desc1))


if __name__ == "__main__":
    unittest.main()
