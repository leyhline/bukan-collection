"""
bukan.keypoint
~~~~~

High-level interface for bukan.py.

Takes one image as input (i.e. a file system path) and produces
a JSON file as output, representing a list of OpenCV KeyPoint objects.
<https://docs.opencv.org/4.2.0/d2/d29/classcv_1_1KeyPoint.html> 

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

import argparse
import json
import sys
import os
from base64 import b64encode
from typing import List, Dict, Any
import cv2 as cv
import numpy as np
from bukan import detect_features, read_image, crop_image


def zip_features(keypoints: List[cv.KeyPoint],
                 descriptors: np.ndarray) -> List[Dict[str, Any]]:
    assert len(keypoints) == descriptors.shape[0]
    features = []
    for i in range(len(keypoints)):
        kp = keypoints[i]
        desc = bytes(descriptors[i, :])
        features.append({
            "x": kp.pt[0],
            "y": kp.pt[1],
            "size": kp.size,
            "angle": kp.angle,
            "response": kp.response,
            "octave": kp.octave,
            "class_id": kp.class_id,
            "base64descriptor": b64encode(desc).decode()
        })
    return features


def main(img_path, out):
    img = read_image(img_path)
    img = crop_image(img)
    keypoints, descriptors = detect_features(img)
    features = zip_features(keypoints, descriptors)
    _, img_filename = os.path.split(img_path)
    fobj = {"filename": img_filename, "features": features}
    json.dump(fobj, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        "Detecting keypoints in images, producing JSON output")
    parser.add_argument("img", type=str, metavar="PATH",
                        help="Path to an image file.")
    parser.add_argument("-o", type=argparse.FileType("x"), default=sys.stdout,
                        metavar="FILENAME", help="Name of output file. (defaults to stdout)")
    args = parser.parse_args()
    main(args.img, args.o)
