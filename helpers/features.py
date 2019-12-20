import os.path
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import time
import numpy.testing as npt
import itertools


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


def log_progress(sequence, every=None, size=None, name='Items'):
    """From: <https://github.com/kuk/log-progress>"""
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


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


def show_image(image, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()


@timeit
def detect_keypoints_and_descriptors(img_paths_df, detector):
    def detect_keypoints(path):
        if not os.path.exists(path):
            return [np.nan, np.nan]
        else:
            img = cv.imread(path)
            img = preprocess(img)
            return detector.detectAndCompute(img, None)
    keypoints = img_paths_df.applymap(detect_keypoints)
    descriptors = keypoints.applymap(operator.itemgetter(1))
    keypoints = keypoints.applymap(operator.itemgetter(0))
    return keypoints, descriptors


def detect_keypoints_and_descriptors_orb(img_paths_df):
    orb = cv.ORB_create()
    return detect_keypoints_and_descriptors(img_paths_df, orb)


def detect_keypoints_and_descriptors_akaze(img_paths_df):
    akaze = cv.AKAZE_create(cv.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, descriptor_size=0, threshold=0.008)
    return detect_keypoints_and_descriptors(img_paths_df, akaze)


def detect_keypoints_and_descriptors_brisk(img_paths_df):
    brisk = cv.BRISK_create(thresh=115)
    return detect_keypoints_and_descriptors(img_paths_df, brisk)


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
def cross_match(descriptors_df, validate=False, norm_type=cv.NORM_HAMMING):
    nr_pages, nr_book_ids = descriptors_df.shape
    nr_distances = 16
    # Allocate memory for all descriptors
    matches = np.full((nr_book_ids, nr_pages, nr_book_ids, nr_pages, nr_distances),
                      np.nan, dtype=np.float32)
    bfmatcher = cv.BFMatcher(norm_type, crossCheck=True)
    distance_getter = operator.attrgetter("distance")
    for query_i, (_, query_pages) in log_progress(enumerate(descriptors_df.items()),
                                                  every=1, size=nr_book_ids, name="Books"):
        for query_j, query_descs in log_progress(enumerate(query_pages),
                                                 every=1, size=nr_pages, name="Pages"):
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


def compare_books(book1_id, book2_id, matches_df):
    select = matches_df.loc[book1_id][book2_id]
    good = []
    bad = []
    for q_page in select.index:
        for t_page in select.loc[q_page].index.levels[0]:
            descs = select.loc[q_page][t_page]
            if q_page == t_page:
                good.append(descs)
            else:
                bad.append(descs)
    good = pd.DataFrame(good)
    bad = pd.DataFrame(bad)
    return good, bad


def threshold_distances(distance_df, threshold):
    """
    Returns the number of pages, where there is at least one useful match.
    A useful match is a match with a distance below the given threshold.
    """
    distance_df = distance_df.dropna(how="all")
    distance_df = distance_df[distance_df < threshold].count(axis=1)
    nr_total = distance_df.size
    nr_positives = distance_df[distance_df > 0].size
    return nr_positives, nr_total


def precision_and_recall(nr_true_positives, nr_false_positives, nr_cond_positives):
    if (nr_true_positives + nr_false_positives) <= 0:
        precision = float("nan")
    else:
        precision = nr_true_positives / (nr_true_positives + nr_false_positives)
    recall = nr_true_positives / nr_cond_positives
    return precision, recall


def precision_recall_curves(matches_df, start, stop, step):
    results = []
    for book1_id, book2_id in itertools.combinations(matches_df.index.levels[0], 2):
        good, bad = compare_books(book1_id, book2_id, matches_df)
        for threshold in np.arange(start, stop, step):
            nr_true_positives, nr_cond_positives = threshold_distances(good, threshold)
            nr_false_positives, nr_cond_negatives = threshold_distances(bad, threshold)
            precision, recall = precision_and_recall(nr_true_positives, nr_false_positives, nr_cond_positives)
            results.append((book1_id, book2_id, threshold, precision, recall))
    return pd.DataFrame(results, columns=["Book1", "Book2", "Threshold", "Precision", "Recall"]).set_index(["Book1", "Book2", "Threshold"])


def precision_recall_intersection(precision_recall_df):
    assert precision_recall_df.size > 1
    first_row_precision, first_row_recall = precision_recall_df.iloc[0]
    assert first_row_precision > first_row_recall
    for thresh, (precision, recall) in precision_recall_df.iterrows():
        if precision < recall:
            thresh_step = thresh - former_thresh
            precision_gradient = (precision - former_precision) / thresh_step
            recall_gradient = (recall - former_recall) / thresh_step
            intersection_thresh = (former_precision - former_recall) / (recall_gradient - precision_gradient)
            intersection_precision = former_precision + intersection_thresh * precision_gradient
            intersection_recall = former_recall + intersection_thresh * recall_gradient
            assert abs(intersection_precision - intersection_recall) < 0.0001
            return thresh + intersection_thresh, intersection_precision
        else:
            former_thresh = thresh
            former_precision = precision
            former_recall = recall


def intersection_df(precision_recall_df):
    results = []
    for book1, book2, thresh in precision_recall_df.index:
        try:
            x_thresh, y_value = precision_recall_intersection(precision_recall_df.loc[book1, book2])
            results.append((book1, book2, x_thresh, y_value))
        except:
            continue
    return pd.DataFrame(results, columns=["Book1", "Book2", "Threshold", "Value"]).set_index(["Book1", "Book2"])
