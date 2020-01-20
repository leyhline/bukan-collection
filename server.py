import pandas as pd
import cv2 as cv
from flask import Flask, render_template, make_response


app = Flask(__name__)
bukan_df = pd.read_parquet("output/matching/concatenated.parquet.gzip", engine="pyarrow")
keypoints_df = pd.read_parquet("output/matching/keypoints.parquet.gzip", engine="pyarrow")
bukan_nr_matches_s = bukan_df["distance"].groupby(['bukan', 'book1', 'page1', 'book2', 'page2']).count().rename("matches")
bukan_nr_matching_pages_s = bukan_nr_matches_s.groupby("bukan").count().rename("pages")


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


@app.route("/")
def index():
    return render_template("index.html", bukans=bukan_nr_matching_pages_s.items())


@app.route("/bukan/<bukan_title>")
def bukan(bukan_title):
    return bukan_nr_matches_s[bukan_title].to_frame().to_html()


@app.route("/bukan/<bukan_title>/<book1_id>/<book1_page>/<book2_id>/<book2_page>")
def bukan_match(bukan_title, book1_id, book1_page, book2_id, book2_page):
    indexer = (bukan_title, int(book1_id), int(book1_page), int(book2_id), int(book2_page))
    matches_df = bukan_df.loc[indexer]
    matches = matches_df_to_list(matches_df)
    img1 = read_image(f"data/{book1_id}/image/{book1_id}_{book1_page:0>5}.jpg")
    img2 = read_image(f"data/{book2_id}/image/{book2_id}_{book2_page:0>5}.jpg")
    keypoints1 = keypoints_df_to_list(keypoints_df.loc[(int(book1_id), int(book1_page))])
    keypoints2 = keypoints_df_to_list(keypoints_df.loc[(int(book2_id), int(book2_page))])
    match_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    enc_result, match_img = cv.imencode(".jpg", match_img)
    assert enc_result
    response = make_response(match_img.tobytes())
    response.content_type = "image/jpeg"
    return response
