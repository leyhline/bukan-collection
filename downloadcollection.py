#!/usr/bin/env python3
"""
A really simple downloader for the full Bukan Collection from the
Center of Open Data in the Humanities (CODH) <http://codh.rois.ac.jp/>.

This just dumps the everything into a `data` folder without any kind
of verification. The folder structure is:

data/{book_id}/data.zip       (for the image archives)
data/{book_id}/manifest.json  (for metadata by IIIF protocol)
"""

__author__ = "Thomas Leyh"
__author_email__ = "thomas.leyh@mailbox.org"
__copyright__ = "(c) 2019 Thomas Leyh"

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlcleanup, urlretrieve


def read_csv():
    with open("bukan-overview.csv", "r") as fd:
        reader = csv.DictReader(fd)
        bukan_list = [row for row in reader if row]
    return bukan_list


def download_dataset(book_id):
    url = f"http://codh.rois.ac.jp/pmjt/book/{book_id}/{book_id}.zip"
    urlretrieve(url, f"data/{book_id}/data.zip")


def download_manifest(book_id):
    url = f"http://codh.rois.ac.jp/pmjt/book/{book_id}/manifest.json"
    urlretrieve(url, f"data/{book_id}/manifest.json")


def process_id(book_id):
    print("Create folder:", book_id)
    try:
        os.mkdir(f"data/{book_id}")
        print("Downloading manifest:", book_id)
        download_manifest(book_id)
        print("Downloading dataset:", book_id)
        download_dataset(book_id)
    except FileExistsError:
        print("Folder already exists:", book_id, "(Skipping download)")


def main():
    print("Let's go boys!")
    try:
        os.mkdir("data")
    except FileExistsError:
        print("Folder already exists: data/")
    bukan_list = read_csv()
    book_ids = [row["国文研書誌ID"] for row in bukan_list]
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(process_id, book_ids)
    finally:
        urlcleanup()


if __name__ == "__main__":
    main()
