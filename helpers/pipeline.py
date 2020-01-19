import cv2 as cv
import numpy as np
import pandas as pd
import time
from itertools import combinations, chain
from math import ceil
from operator import itemgetter
from collections.abc import Iterable


def timeit(function):
    """
    This is a decorator for measuring execution time of a function.
    Returns a tuple: (time in seconds, original function's return value)
    """
    def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        value = function(*args, **kwargs)
        stop_time = time.monotonic()
        return stop_time - start_time, value
    return wrapper


def build_page_index(offset_df):
    index = []
    for book1_id, book2_id in combinations(offset_df.columns, 2):
        for page in offset_df.index:
            index.append((book1_id, book2_id, page))
    return pd.MultiIndex.from_tuples(index, names=("Book1", "Book2", "Page"))


def build_page_df(offset_df, radius_factor=0.025):
    """
    Build a DataFrame where each row is a Book1, Book2, Book1-Page combination.
    Each column is an offset, specifying the the corresponding Book2-Page
    for later comparison. If this is 0, then no corresponding page exists.
    """
    index = build_page_index(offset_df)
    nr_pages = offset_df.index.size
    def handle_borders(n):
        if n < 0:
            return 0
        elif n > nr_pages:
            return 0
        else:
            return n
    radius = ceil(nr_pages * radius_factor)
    diameter = radius * 2 + 1
    page_array = np.empty((len(index), diameter), dtype=np.int32)
    for i, (_, _, page) in enumerate(index):
        first_page = page - radius
        last_page = page + radius
        pages = np.fromiter((handle_borders(p) for p in range(first_page, last_page + 1)),
                            count=diameter, dtype=np.int32)
        page_array[i,:] = pages
    columns = pd.RangeIndex(-radius, radius+1)
    return pd.DataFrame(page_array, index=index, columns=columns)


def crop_image(img, horizontal_factor=0.1, vertical_factor=0.15):
    height, width = img.shape
    return img[int(height * vertical_factor) : int(height - height * vertical_factor),
               int(width * horizontal_factor): int(width - width * horizontal_factor)]


def read_image(path):
    img = cv.imread(path, flags=cv.IMREAD_REDUCED_GRAYSCALE_4)
    img = crop_image(img)
    return img


@timeit
def detect_and_extract(path_df, detector):
    def process_path(path):
        if not path:
            return (np.nan, np.nan)
        else:
            return detector.detectAndCompute(read_image(path), None)
    keypoints = path_df.applymap(process_path)
    descriptors = keypoints.applymap(itemgetter(1))
    keypoints = keypoints.applymap(itemgetter(0))
    return keypoints, descriptors


@timeit
def applymapi(dataframe, func, **kwargs):
    """
    Applies a function elementwise on a pandas.DataFrame.
    It's similar to the pandas.DataFrame.applymap method
    but here the function also takes the index and the value
    (instead of just the value).
    
    The env argument is a dictionary holding objects
    that might be needed by the elementwise func.
    """
    def series_func(series):
        book1, book2, page_book1 = series.name
        new_values = np.empty_like(series.values, dtype=np.object)
        for i, (page_offset, value) in enumerate(series.items()):
            new_values[i] = func(book1, book2, page_book1, page_offset, value, kwargs)
        return new_values
    return dataframe.transform(series_func, axis=1)


def find_matches(book1, book2, page_book1, page_offset, page_book2, env):
    if page_book2 == 0:
        return np.nan
    else:
        try:
            desc_book1 = env["descriptors"][book1][page_book1]
        except KeyError:
            #print(f"No features for book {book1}, page {page_book1}: Returning empty list.")
            return []
        try:
            desc_book2 = env["descriptors"][book2][page_book2]
        except KeyError:
            #print(f"No features for book {book2}, page {page_book2}: Returning empty list.")
            return []
        if isinstance(desc_book1, np.ndarray) and isinstance(desc_book2, np.ndarray):
            matches = env["matcher"].radiusMatch(desc_book1, desc_book2, env["max_distance"], compactResult=True)
            return list(chain.from_iterable(matches))
        else:
            return np.nan


def select_keypoints(book1, book2, page_book1, page_offset, page_book2, env):
    matches = env["matches"].loc[(book1, book2, page_book1), page_offset]
    if not isinstance(matches, Iterable):
        return (np.nan, np.nan)
    else:
        points1 = []
        points2 = []
        mask = np.zeros(len(matches), dtype=np.bool)
        points1_generator = (env["keypoints"][book1][page_book1][match.queryIdx].pt for match in matches)
        points2_generator = (env["keypoints"][book2][page_book2][match.trainIdx].pt for match in matches)
        points_generator = zip(points1_generator, points2_generator)
        for j, (point1, point2) in enumerate(points_generator):
            if env["filter"](point1, point2):
                points1.append(point1)
                points2.append(point2)
                mask[j] = True
        assert len(points1) == len(points2)
        return (np.array(points1, dtype=np.float32), np.array(points2, dtype=np.float32)), mask


@timeit
def compute_homography_and_mask(selected_keypoints):
    def find_homography(keypoints):
        if isinstance(keypoints, Iterable):
            points1, points2 = keypoints
            if (points1.size == 0) and (points2.size == 0):
                na_homography = np.full((3, 3), np.nan, dtype=np.float64)
                empty_mask = np.empty((0, 1), dtype=np.uint8)
                return (na_homography, empty_mask)
            else:
                homography, mask = cv.findHomography(points1, points2, cv.RANSAC)
                return homography, np.squeeze(mask).astype(np.bool)
        else:
            return (np.nan, np.nan)
    results = selected_keypoints.applymap(find_homography)
    homographies = results.applymap(itemgetter(0))
    masks = results.applymap(itemgetter(1))
    return homographies, masks


def sum_homography_mask(mask_df):
    return mask_df.applymap(lambda x: x.sum() if isinstance(x, np.ndarray) else np.nan)


def calculate_metrics(result_df, thresholds):
    positives = result_df[0].dropna()
    nr_positives = positives.size
    negatives = result_df.drop(0, axis=1).stack()
    metrics = np.empty((len(thresholds), 3), dtype=np.float64)
    for i, threshold in enumerate(thresholds):
        nr_true_positives = (positives > threshold).sum()
        nr_false_positives = (negatives > threshold).sum()
        precision = nr_true_positives / (nr_true_positives + nr_false_positives)
        recall = nr_true_positives / nr_positives
        f1 = (2 * precision * recall) / (precision + recall)
        metrics[i, 0] = precision
        metrics[i, 1] = recall
        metrics[i, 2] = f1
    return pd.DataFrame(metrics, columns=["Precision", "Recall", "F1"], index=thresholds)


def is_good_homography(homography):
    if isinstance(homography, np.ndarray):
        return (not np.any(np.isnan(homography))) and (not np.any(np.absolute(homography[2,:2]) > 0.001))
    else:
        return False

@timeit
def filter_out_bad_homographies(homographies, masks):
    good_homographies = homographies.applymap(is_good_homography)
    return masks.where(good_homographies)
