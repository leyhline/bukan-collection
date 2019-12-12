import os.path
import cv2 as cv
import numpy as np
import pandas as pd
import operator
import time


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


def read_offset_csv(path):
    """
    Reads a CSV with book offsets and returns a pandas DataFrame of int8 values.
    The header are the IDs of the corresponding books.
    If pages are missing these are indicated by value -128 (min int8).
    """
    df = pd.read_csv(path, header=0, na_values=" ").fillna(-128).astype(np.int8)
    df.index = df.index + 1
    return df


def transform_offsets_to_paths(offsets_df):
    def col_to_paths(column):
        """Transforms a column of offsets to page numbers."""
        page_numbers = []
        current_page = 0
        for x in column:
            if x == -128:
                current_page += 1
                page_numbers.append("")
            else:
                current_page += 1 + x
                page_numbers.append(f"data/{column.name}/image/{column.name}_{current_page:>05}.jpg")
        return page_numbers
    return offsets_df.transform(col_to_paths, axis=0)


def preprocess(img):
    """
    Convert to grayscale, crop and resize to 25%.
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = img.shape
    crop_horizontal_factor = 0.1
    crop_vertical_factor = 0.15
    img = img[int(height * crop_vertical_factor) : int(height  - height * crop_vertical_factor),
              int(width * crop_horizontal_factor): int(width - width * crop_horizontal_factor)]
    scale_factor = 4
    img = cv.resize(img, (width // scale_factor, height // scale_factor),
                    interpolation=cv.INTER_AREA)
    return img


def detect_keypoints_and_descriptors(img_paths_df):
    orb = cv.ORB_create()
    def detect_keypoints(path):
        if not os.path.exists(path):
            return [np.nan, np.nan]
        else:
            img = cv.imread(path)
            img = preprocess(img)
            return orb.detectAndCompute(img, None)
    keypoints = img_paths_df.applymap(detect_keypoints)
    descriptors = keypoints.applymap(operator.itemgetter(1))
    keypoints = keypoints.applymap(operator.itemgetter(0))
    return keypoints, descriptors


def validate_cross_matches(matches_dataframe, original_ndarray):
    """Raises an exception if reshaping (or a previous operation) went wrong."""
    nr_query_books, nr_query_pages, nr_train_books, nr_train_pages, nr_descs = original_ndarray.shape
    for query_book_id in range(nr_query_books):
        for query_page in range(nr_query_pages):
            for train_book_id in range(nr_train_books):
                for train_page in range(nr_train_pages):
                    x = query_book_id * nr_query_pages + query_page
                    y = (train_book_id * nr_train_pages + train_page) * nr_descs
                    npt.assert_array_equal(
                        matches_dataframe.iloc[x, y:y+nr_descs],
                        original_ndarray[query_book_id, query_page, train_book_id, train_page])


@timeit
def cross_match(descriptors_df, validate=False):
    nr_pages, nr_book_ids = descriptors_df.shape
    nr_distances = 16
    # Allocate memory for all descriptors
    matches = np.full((nr_book_ids, nr_pages, nr_book_ids, nr_pages, nr_distances),
                      np.nan, dtype=np.float32)
    bfmatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    distance_getter = operator.attrgetter("distance")
    for query_i, (_, query_pages) in enumerate(descriptors_df.items()):
        for query_j, query_descs in enumerate(query_pages):
            for train_i, (_, train_pages) in enumerate(descriptors_df.items()):
                for train_j, train_descs in enumerate(train_pages):
                    if (not isinstance(query_descs, np.ndarray) or
                        not isinstance(train_descs, np.ndarray)):
                        continue
                    else:
                        match_list = bfmatcher.match(query_descs, train_descs)
                        match_list = np.fromiter(map(distance_getter, match_list),
                                                 dtype=np.float32, count=len(match_list))
                        match_list.sort()
                        match_list = match_list[:nr_distances]
                    np.copyto(matches[query_i, query_j, train_i, train_j, 0:(match_list.size)],
                              match_list, casting="no")
    index_rows = pd.MultiIndex.from_product((descriptors_df.columns, descriptors_df.index))
    index_cols = pd.MultiIndex.from_product((descriptors_df.columns, descriptors_df.index, pd.RangeIndex(nr_distances)))  
    matches_df = pd.DataFrame(matches.reshape((nr_book_ids * nr_pages, nr_book_ids * nr_pages * nr_distances)),
                              index=index_rows, columns=index_cols)
    if validate:
        validate_cross_matches(matches_df, matches)
    return matches_df
