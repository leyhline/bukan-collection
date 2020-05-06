# Detection of Differences between Printed Pages and its Application on Bukan

This is a project I worked on during my internship at the National Institute of Informatics in Tokyo, Japan. Essentially, it is about the application of classical Computer Vision on historic japanese literature, thus classifying as *Digital Humanitites*.

Bukan are books from Japan's Edo Period (1603-1868), listing people of influence together with crests, family trees etc. These books were bestsellers, printed using woodblocks. As a result, there is a large number of prints and editions, hiding away potential useful information for the humanities scholar. To lessen the burden on the human researcher a computer might help in comparing the pages, showing recommendations and visualizations.

By utilizing proved techniques from Computer Science and Computer Vision—most notably [Feature Detection](https://en.wikipedia.org/wiki/Feature_detection_(computer_vision)) and [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)—a database is populated with matching pages between different prints. This approach has an accuracy of above 95% when looking for the same page in a different print of a book. Furthermore, this can be used for creating a nice-looking overlay of a page-pair, thus resulting in a useful visualization to quickly discern the differences.

For more details, just have a look at my [Internship Report](InternshipReport.pdf) where I also included a few graphics and examples.

## Structure of this Repository

I used this repository mainly for running experiments. Obviously, I was not quite sure about my approach, its performance and the usefulness of the results, which reflects in the messy structure and the large number of Jupyter notebooks.

## Dependencies

First you need to download the data. Next you need to install the Python dependencies (recommended: in a virtualenvironment).

### Downloading the Dataset: Bukan Collection

The data is publicly available on the servers of the [Center of Open Data in the Humanities](http://codh.rois.ac.jp/). I have a small dirty script prepared for downloading everything (around 200GB). So it will take some time depending on your internet connection. Just run:

```
python3 downloadcollection.py
```

Then wait for some hours and pray it will not get interrupted since I do not catch this. (but a logfile is created: `downloadcollection.log`) The data is stored in a newly created `data` folder.

### Installing Python Dependencies

This is *optional* but I recommend creating a Virtual Environment first. There are multiple ways to do this (pipenv, conda…) but basically it is unter Linux:

```
python3 -m venv venv
source venv/bin/activate
```

Next, you can simply install all the dependencies (mostly scientific libraries) via:

```
pip3 install -r requirements.txt
```

Now you are ready to go to run the code, preferrably by opening a Jupyter Notebook:

```
jupyter notebook
```

## Webserver

There is also code for a simple demo application using [Flask](https://flask.palletsprojects.com/). This depends on some data that is created by running the notebooks and I am currently not sure how to release this data since it is quite a lot. I hope I can find ressources for running the application myself somewhere. Nevertheless, this is the code for starting the development webserver:

```
export FLASK_APP=server.py
export FLASK_ENV=development  # optional
flask run
```