import os.path
import json
import pandas as pd
import cv2 as cv
from flask import Flask, render_template, make_response


bukan_overview_path = "output/grey/overview.parquet"
assert os.path.exists(bukan_overview_path)
overview_df = pd.read_parquet(bukan_overview_path)
bukan_title_df = overview_df.groupby(["Title", "TitleHiragana", "TitleRomanji"]).count()["Release"]
bukan_data = {}
image_width = 990
image_height = 660


app = Flask(__name__)


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
    title = bukan_title_df.index[bukan_title_df.index.get_level_values("TitleRomanji") == bukan_title][0]
    selection = matches.droplevel(0).loc[(int(lr), int(book1_id), int(book1_page), int(book2_id), int(book2_page))]
    dst_points = selection[["dst_x", "dst_y", "dst_size"]].values
    json_dst_points = json.dumps(dst_points.tolist())
    indexer = (bukan_title, book1_id, book1_page, book2_id, book2_page, lr)
    return render_template("image.html",
                           indexer=indexer, title=title, keypoints=json_dst_points,
                           width=image_width if lr == "0" else image_width // 2, height=image_height)


@app.route("/image/<book_id>/<page>/<lr>")
def image(book_id, page, lr):
    image = cv.imread(f"data/grey/{book_id}/image/{book_id}_{page:0>5}_{lr}.jpg", cv.IMREAD_GRAYSCALE)
    enc_result, jpg_image = cv.imencode(".jpg", image)
    assert enc_result
    response = make_response(jpg_image.tobytes())
    response.content_type = "image/jpeg"
    return response


@app.route("/matching/<bukan_title>/<book1_id>/<book1_page>/<book2_id>/<book2_page>/<lr>")
def matching(bukan_title, book1_id, book1_page, book2_id, book2_page, lr):
    if bukan_title in bukan_data:
        matches = bukan_data[bukan_title]
    else:
        matches = pd.read_parquet(f"output/grey/{bukan_title}.parquet")
        bukan_data[bukan_title] = matches
    src_image = cv.imread(f"data/grey/{book1_id}/image/{book1_id}_{book1_page:0>5}_{lr}.jpg", cv.IMREAD_GRAYSCALE)
    dst_image = cv.imread(f"data/grey/{book2_id}/image/{book2_id}_{book2_page:0>5}_{lr}.jpg", cv.IMREAD_GRAYSCALE)
    selection = matches.droplevel(0).loc[(int(lr), int(book1_id), int(book1_page), int(book2_id), int(book2_page))]
    src_points = selection[["src_x", "src_y"]].values
    dst_points = selection[["dst_x", "dst_y"]].values
    homography, _ = cv.findHomography(src_points, dst_points, 0)
    dst_height, dst_width = dst_image.shape
    src_image_warped = cv.warpPerspective(src_image, homography, (dst_width, dst_height))
    enc_result, src_image_warped_jpg = cv.imencode(".jpg", src_image_warped)
    assert enc_result
    response = make_response(src_image_warped_jpg.tobytes())
    response.content_type = "image/jpeg"
    return response
