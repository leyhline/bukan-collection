import os.path
import pandas as pd
import cv2 as cv
from flask import Flask, render_template, make_response


bukan_data_path = "output/matching/concatenated.parquet.gzip"
assert os.path.exists(bukan_data_path)
keypoints_path = "output/matching/keypoints.parquet.gzip"
assert os.path.exists(keypoints_path)
bukan_df = pd.read_parquet(bukan_data_path, engine="pyarrow")
matches_df = bukan_df.droplevel(0).sort_index()
keypoints_df = pd.read_parquet(keypoints_path, engine="pyarrow")
bukan_nr_matches_s = bukan_df["distance"].groupby(['bukan', 'book1', 'page1', 'book2', 'page2']).count().rename("matches")
bukan_nr_matching_pages_s = bukan_nr_matches_s.groupby("bukan").count().rename("pages")


def create_matching_book_dict(matches_df):
    matching_ids_table = matches_df.index.droplevel([1, 3, 4]).drop_duplicates()
    matching_ids = {}
    for book1_id, book2_id in matching_ids_table:
        if book1_id in matching_ids:
            matching_ids[book1_id].add(book2_id)
        else:
            matching_ids[book1_id] = {book2_id}
        if book2_id in matching_ids:
            matching_ids[book2_id].add(book1_id)
        else:
            matching_ids[book2_id] = {book1_id}
    return matching_ids


matching_ids = create_matching_book_dict(matches_df)
app = Flask(__name__)


def crop_image(img, horizontal_factor=0.1, vertical_factor=0.15):
    height, width = img.shape
    return img[int(height * vertical_factor) : int(height - height * vertical_factor),
               int(width * horizontal_factor): int(width - width * horizontal_factor)]


def read_image(path):
    img = cv.imread(path, flags=cv.IMREAD_REDUCED_GRAYSCALE_4)
    img = crop_image(img)
    return img


def matches_df_to_list(matches_df):
    return [cv.DMatch(*args) for args
            in zip(*(column for _, column in matches_df.items()))]


def keypoints_df_to_list(keypoints_df):
    return [cv.KeyPoint(*args) for args
            in zip(*(column for _, column in keypoints_df.items()))]


def create_match_image(book1_id, book1_page, book2_id, book2_page, matches):
    img1 = read_image(f"data/{book1_id}/image/{book1_id}_{book1_page:0>5}.jpg")
    img2 = read_image(f"data/{book2_id}/image/{book2_id}_{book2_page:0>5}.jpg")
    keypoints1 = keypoints_df_to_list(keypoints_df.loc[(int(book1_id), int(book1_page))])
    keypoints2 = keypoints_df_to_list(keypoints_df.loc[(int(book2_id), int(book2_page))])
    match_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    enc_result, match_img = cv.imencode(".jpg", match_img)
    assert enc_result
    return match_img.tobytes()


@app.route("/")
def index():
    return render_template("index.html", bukans=bukan_nr_matching_pages_s.items())


@app.route("/bukan/<bukan_title>")
def bukan(bukan_title):
    nr_matches = bukan_nr_matches_s[bukan_title]
    return render_template("bukan.html", matches=nr_matches.items())


@app.route("/matches/<book1_id>/<book1_page>/<book2_id>/<book2_page>")
def matches(book1_id, book1_page, book2_id, book2_page):
    indexer = (int(book1_id), int(book1_page), int(book2_id), int(book2_page))
    page_matches_df = matches_df.loc[indexer]
    matches = matches_df_to_list(page_matches_df)
    match_img = create_match_image(book1_id, book1_page, book2_id, book2_page, matches)
    response = make_response(match_img)
    response.content_type = "image/jpeg"
    return response


@app.route("/book/<book_id>")
def book(book_id):
    book_id_set = matching_ids[int(book_id)]
    return render_template("book.html", book_ids=book_id_set)
