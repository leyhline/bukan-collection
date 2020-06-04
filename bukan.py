"""
bukan
~~~~~

Essential functions for comparing two printed pages.
Depends on the Python bindings of OpenCV <https://opencv.org/>.
For details, see the corresponding paper and GitHub repository:
<https://github.com/leyhline/bukan-collection>

:copyright: (c) 2020 Thomas Leyh <leyht@informatik.uni-freiburg.de>
:licence: GPLv3, see LICENSE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import cv2 as cv
from typing import Tuple, List
from itertools import chain, compress
import argparse


DETECTOR_AKAZE = cv.AKAZE_create(cv.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, descriptor_size=0, threshold=0.005)
"""Default feature detector for `detect_features` (see corresponding paper for details)."""
HAMMING_MATCHER = cv.BFMatcher_create(normType=cv.NORM_HAMMING)
"""When matching AKAZE features, this calculates the hamming distance between descriptors."""


def read_image(path: os.PathLike) -> np.ndarray:
    """
    Using OpenCV to read the image at `path` into memory.
    This directly returns the greyscaled image, even if it is originally
    in color and it scales down both width and height to 25%.
    Thus, the resulting `ndarray` will be 2-dimensional with shape
    (height, width).
    """
    if os.path.exists(path):
        img = cv.imread(path, flags=cv.IMREAD_REDUCED_GRAYSCALE_4)
        if img is not None:
            return img
        else:
            raise RuntimeError("Not an image file: %s" % path)
    else:
        raise FileNotFoundError(path)


def crop_image(img: np.ndarray, target_height=660, target_width=990) -> np.ndarray:
    """
    Returns a centered crop of `img` of shape (`target_height`, `target_width`).
    The default values are something I specifically chose for the Bukan Collection
    to speed up the keypoint detection. I am not sure if cropping is necessary
    in the first place.
    """
    height, width = img.shape
    x1 = (width - target_width) // 2
    y1 = (height - target_height) // 2
    x2 = x1 + target_width
    y2 = y1 + target_height
    return img[y1:y2, x1:x2]


def split_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split `img` at its horizontal center, returning left and right part.
    This is for images showing both left and right page at once.
    """
    _, width = img.shape
    half_width = width // 2
    return img[:, :half_width], img[:, half_width:]


def detect_features(img: np.ndarray, detector: cv.Feature2D=DETECTOR_AKAZE
                    ) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    """
    Detects keypoints and computes descriptors for the given `img` using
    `detector`. By default, `DETECTOR_AKAZE` is used as in the original paper.
    If detection was successful, a list of KeyPoint objects
    <https://docs.opencv.org/4.2.0/d2/d29/classcv_1_1KeyPoint.html>
    and a numpy array of shape (len(keypoints), ...) for the descriptors
    is returned. For AKAZE, the array will be of type `np.uint8`.
    """
    keypoints, descriptors = detector.detectAndCompute(img, None)
    if len(keypoints) > 0:
        return keypoints, descriptors
    else:
        raise Exception("No features found in given image")


def is_near(points_pair: Tuple[Tuple[float], Tuple[float]],
            wradius = 100., hradius = 100.) -> bool:
    """
    Helper function used in `match_pair` for checking if the given coordinates
    are close to each other. 
    """
    (x1, y1), (x2, y2) = points_pair
    return ((x1 - wradius) <= x2 <= (x1 + wradius)) and ((y1 - hradius) <= y2 <= (y1 + hradius))


def match_pair(keypoints1: List[cv.KeyPoint],
               descriptors1: np.ndarray,
               keypoints2: List[cv.KeyPoint],
               descriptors2: np.ndarray,
               matcher: cv.DescriptorMatcher=HAMMING_MATCHER,
               max_match_distance=100) -> Tuple[List[cv.DMatch], np.ndarray]:
    """
    Matches a pair of features against each other, consisting of `keypoints1`
    and `descriptors1` on the one side and `keypoints2` and `descriptors2` on
    the other. For `DETECTOR_AKAZE` the default `HAMMING_MATCHER` must be used.
    If the features came from a different detector, check the OpenCV documentation.
    <https://docs.opencv.org/4.2.0/d8/d9b/group__features2d__match.html>
    The function returns a list of DMatch objects
    <https://docs.opencv.org/4.2.0/d4/de0/classcv_1_1DMatch.html> and
    an 3x3 homography matrix that may be used with OpenCV's `warpPerspective`.  
    """
    assert len(keypoints1) == descriptors1.shape[0]
    assert len(keypoints2) == descriptors2.shape[0]
    # 1. find matches
    matches = matcher.radiusMatch(descriptors1, descriptors2, max_match_distance)
    matches = list(chain.from_iterable(matches))
    if len(matches) == 0:
        raise Exception("No matches found")
    # 2. select keypoints
    points1 = np.empty((len(matches), 2), dtype=np.float32)
    points2 = np.empty((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # 3. remove matches that are too far away from each other by applying a boolean mask
    zipped_points = zip(points1, points2)
    points_mask = np.fromiter(map(is_near, zipped_points), dtype=np.bool, count=len(matches))
    points1 = points1[points_mask]
    points2 = points2[points_mask]
    assert points1.shape == points2.shape
    if points1.size == 0:
        raise Exception("All matches are too far away from each other")
    # 4. compute homography using RANSAC
    h, h_mask = cv.findHomography(points1, points2, cv.RANSAC)
    if not h_mask.sum() > 0:
        raise Exception("No homography found between images")
    # 5. check for really bad homographies, setting is_bad_h to true
    is_bad_h = np.any(np.isnan(h)) or np.any(np.absolute(h[2,:2]) > 0.001)
    if is_bad_h:
        raise Exception("Perspective change of homography is too large")
    # 6. select only relevant matches by 'reversing' all operations above
    h_mask = np.squeeze(h_mask).astype(np.bool)
    j = 0
    for i in range(len(points_mask)):
        if points_mask[i]:
            if not h_mask[j]:
                points_mask[i] = False
            j += 1
    assert points_mask.sum() == h_mask.sum()
    relevant_matches = list(compress(matches, points_mask))
    return relevant_matches, h


def main(img_paths, overlay):
    """This is mainly for demo purposes when running from the command line."""
    if len(img_paths) == 1:
        path = img_paths[0]
        img = read_image(path)
        keypoints, _ = detect_features(img)
        outimg = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print("Showing %d keypoints of %s" % (len(keypoints), path))
        cv.imshow("Keypoints: %d" % len(keypoints), outimg)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        path1 = img_paths[0]
        path2 = img_paths[1]
        img1 = read_image(path1)
        img2 = read_image(path2)
        keypoints1, descriptors1 = detect_features(img1)
        keypoints2, descriptors2 = detect_features(img2)
        matches, h = match_pair(keypoints1, descriptors1, keypoints2, descriptors2)
        if overlay:
            height, width = img2.shape
            img1_warped = cv.warpPerspective(img1, h, (width, height))
            outimg = np.empty((img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
            outimg[:,:,0] = img1_warped
            outimg[:,:,1] = img2
            outimg[:,:,2] = (img1_warped + img2) // 2
        else:
            outimg = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print("Showing %d matches between %s and %s" % (len(matches), path1, path2))
        cv.imshow("Matching Keypoints: %d" % len(matches), outimg)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test image matching from the command line.")
    parser.add_argument("img", nargs="+", type=str,
                        help="One or better two image files.", metavar="PATH")
    parser.add_argument("-o", action="store_true", help="show image overlay if homography exists.")
    args = parser.parse_args()
    main(args.img, args.o)
