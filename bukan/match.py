"""
bukan.match
~~~~~

High-level interface for bukan.py.

Matches the keypoints of two books which are represented as JSON files.
(see keypoint.py) 

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
import sys
import argparse
import gzip
import json
from typing import Dict, Any, List, Tuple
from base64 import b64decode
from itertools import chain
import cv2 as cv
import numpy as np
from bukan import match_pair


AKAZE_DESCRIPTOR_LENGTH = 61
Book = List[Tuple[str, List[cv.KeyPoint], np.ndarray]]


def encode_features(features: List[Dict[str, Any]]) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    """
    Takes a list of features as returned from bukan.keypoint and converts
    the JSON representation back to a list of `cv.KeyPoint` objects and
    a numpy 2D-array for the descriptors.
    """
    keypoints = []
    descriptors = np.empty((len(features), AKAZE_DESCRIPTOR_LENGTH), dtype=np.uint8)
    for i, f in enumerate(features):
        kp = cv.KeyPoint(f["x"], f["y"], f["size"], f["angle"], f["response"], f["octave"], f["class_id"])
        keypoints.append(kp)
        desc = b64decode(f["base64descriptor"])
        assert len(desc) == AKAZE_DESCRIPTOR_LENGTH
        descriptors[i, :] = np.frombuffer(desc, dtype=np.uint8)
    return keypoints, descriptors


def book_to_images(book: Dict[str, Any]) -> Book:
    """
    Takes a dictionary as returned from `bukan.keypoint` and returns a
    list of 3-tuples, each one holding these entries:
    1. The image's filename (the features' source)
    2. A list of `cv.KeyPoint` objects
    3. A numpy 2D-array holding the descriptors (type: uint8) 
    """
    features = []
    for image in book["images"]:
        filename = image["filename"]
        keypoints, descriptors = encode_features(image["features"])
        features.append((filename, keypoints, descriptors))
    return features


def read_json(path: os.PathLike) -> Dict[str, Any]:
    _, ext = os.path.splitext(path)
    if ext == ".gz":
        with gzip.open(path, "rb") as fd:
            content = json.load(fd)
    else:
        with open(path, "r") as fd:
            content = json.load(fd)
    return content


def calculate_matches(book1: Book, book2: Book, threshold,
                      verbose=True) -> List[Dict[str, any]]:
    """
    Takes two lists with image features as returned from `book_to_images`
    and matches all features against each other using `bukan.match_pair`.
    This takes quadratic time and returns a list with matches where
    the score (the number of matching features) is above a specific threshold.
    Each match is a dictionary. 
    """
    counter = 0
    counter_exceptions = 0
    total = len(book1) * len(book2)
    match_results = []
    for filename1, keypoints1, descriptors1 in book1:
        for filename2, keypoints2, descriptors2 in book2:
            try:
                matches, h = match_pair(keypoints1, descriptors1, keypoints2, descriptors2)
                score = len(matches)
                if score > threshold:
                    match_results.append({
                        "file1": filename1,
                        "file2": filename2,
                        "score": score,
                        "homography": list(chain.from_iterable(h))
                    })
            except Exception:
                counter_exceptions += 1
            counter += 1
            if verbose and (counter % 100) == 0:
                sys.stdout.write(f"\rProcessed pairs: {counter} / {total}")
                sys.stdout.flush()
    if verbose:
        sys.stdout.write("\n")
        print(f"Pairs without match: {counter_exceptions}")
    return match_results


def main(book1_path, book2_path, threshold, out, do_compress):
    book1 = read_json(book1_path)
    book1 = book_to_images(book1)
    book2 = read_json(book2_path)
    book2 = book_to_images(book2)
    matches = calculate_matches(book1, book2, threshold)
    results = {
        "path1": book1_path,
        "path2": book2_path,
        "threshold:": threshold,
        "matches": matches}
    if do_compress:
        results = json.dumps(results).encode()
        out += ".gz"
        with gzip.open(out, "xb") as fd:
            fd.write(results)
    else:
        with open(out, "x") as fd:
            json.dump(results, fd)
    print(f"{len(matches)} matches written to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matching keypoints of two books given as JSON file.")
    parser.add_argument("book1", metavar="PATH",
                        help="JSON file of first book.")
    parser.add_argument("book2", metavar="PATH",
                        help="JSON file of second book")
    parser.add_argument("-t", "--threshold", type=int, default=40,
                        help="Thresholding the score of matches. (default: 40)")
    parser.add_argument("-o", "--output", metavar="FILENAME", default="match.json",
                        help="Name of output file. (default: match.json)")
    parser.add_argument("-C", "--compress", action="store_true", help="Enable gzip compression.")
    args = parser.parse_args()
    main(args.book1, args.book2, args.threshold, args.output, args.compress)
