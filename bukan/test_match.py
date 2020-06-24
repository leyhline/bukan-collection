import unittest
from match import book_to_images, encode_features, calculate_matches
import cv2 as cv
import numpy as np


BOOK_KEYPOINT_DICT = {
    "path": "200019472/image/",
    "images": [
        {
            "filename": "200019472_00097.jpg",
            "features": [
                {
                    "x": 108.44136047363281,
                    "y": 175.09744262695312,
                    "size": 4.800000190734863,
                    "angle": 259.07855224609375,
                    "response": 0.0064725857228040695,
                    "octave": 0,
                    "class_id": 0,
                    "base64descriptor": "8aw9PHwFgyaevh9/QOT70z/+8Sd/+vAGH/RJUAAL8kQXZEjOf/RJnxI0QCfAgiz5sz9/8bdAn2jf/v5tMg=="
                },
                {
                    "x": 62.79212951660156,
                    "y": 175.51565551757812,
                    "size": 4.800000190734863,
                    "angle": 359.8970031738281,
                    "response": 0.00845060870051384,
                    "octave": 0,
                    "class_id": 0,
                    "base64descriptor": "Y9mz/O0Evzz99n57lwPmidt/J3v7a/4dzOUv8gbm0mfP5m9/38EU7DYwBRqSLdmAKdC0l79d+i57ktfvGw=="
                }
            ]
        }
    ]
}


class TestMatch(unittest.TestCase):
    def test_book_to_images(self):
        images = book_to_images(BOOK_KEYPOINT_DICT)
        self.assertEqual(len(images), 1)
        filename, keypoints, descriptors = images[0]
        self.assertEqual(filename, "200019472_00097.jpg")
        self.assertEqual(len(keypoints), 2)
        self.assertIsInstance(keypoints[0], cv.KeyPoint)
        self.assertEqual(descriptors.shape, (2, 61)),
        self.assertEqual(descriptors.dtype, np.uint8)

    def test_encode_features(self):
        features = BOOK_KEYPOINT_DICT["images"][0]["features"]
        keypoints, descriptors = encode_features(features)
        self.assertEqual(len(keypoints), 2)
        self.assertIsInstance(keypoints[0], cv.KeyPoint)
        self.assertEqual(descriptors.shape, (2, 61)),
        self.assertEqual(descriptors.dtype, np.uint8)

    def test_calculate_matches(self):
        book = book_to_images(BOOK_KEYPOINT_DICT)
        matches = calculate_matches(book, book, 40, False)
        self.assertEqual(len(matches), 0)  # Because two keypoints are not enough.
