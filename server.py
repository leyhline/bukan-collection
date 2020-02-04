import os.path
import pandas as pd
import cv2 as cv
from flask import Flask, render_template, make_response


bukan_overview_path = "output/grey/overview.parquet"
assert os.path.exists(bukan_overview_path)
overview_df = pd.read_parquet(bukan_overview_path)
bukan_title_df = overview_df.groupby(["Title", "TitleHiragana", "TitleRomanji"]).count()["Release"]
bukan_data = {}

app = Flask(__name__)


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
    return render_template("index.html", bukans=bukan_title_df.items())


@app.route("/bukan/<bukan_title>")
def bukan(bukan_title):
    if bukan_title in bukan_data:
        matches = bukan_data[bukan_title]
    else:
        matches = pd.read_parquet(f"output/grey/{bukan_title}.parquet")
        bukan_data[bukan_title] = matches
    matches_index = (matches
        .index.droplevel(["title","match"])
        .unique()
        .reorder_levels(["book1", "page1", "book2", "page2", "lr"])
        .sort_values())
    title = bukan_title_df.index[bukan_title_df.index.get_level_values("TitleRomanji") == bukan_title][0]
    return render_template("bukan.html", matches=matches_index, title=title)


@app.route("/bukan/<bukan_title>/<book1_id>/<book1_page>/<book2_id>/<book2_page>/<lr>")
def matches(bukan_title, book1_id, book1_page, book2_id, book2_page, lr):
    if bukan_title in bukan_data:
        matches = bukan_data[bukan_title]
    else:
        matches = pd.read_parquet(f"output/grey/{bukan_title}.parquet")
        bukan_data[bukan_title] = matches


    indexer = (int(book1_id), int(book1_page), int(book2_id), int(book2_page))
    page_matches_df = matches_df.loc[indexer]
    matches = matches_df_to_list(page_matches_df)
    match_img = create_match_image(book1_id, book1_page, book2_id, book2_page, matches)
    response = make_response(match_img)
    response.content_type = "image/jpeg"
    return response


if __name__ == "__main__":
    app.run(threaded=False)
