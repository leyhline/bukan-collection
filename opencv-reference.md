# OpenCV Reference
Especially for the Python bindings.

## Image Processing

<https://docs.opencv.org/4.1.1/d2/d96/tutorial_py_table_of_contents_imgproc.html>

* `cv.imread(path, flags=cv.IMREAD_COLOR)`
  <https://docs.opencv.org/4.1.1/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>
* `cv.cvtColor(image, cv.COLOR_BGR2GRAY)`
  <https://docs.opencv.org/4.1.1/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab>

## Feature Detection

<https://docs.opencv.org/4.1.1/d9/d97/tutorial_table_of_content_features2d.html>
<https://docs.opencv.org/4.1.1/db/d27/tutorial_py_table_of_contents_feature2d.html>

* `cv.cornerHarris(image, 2, 3, 0.04)`
  <https://docs.opencv.org/4.1.1/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345>
* `cv.goodFeaturesToTrack(gray,25,0.01,10)`
  <https://docs.opencv.org/4.1.1/dd/d1a/group__imgproc__feature.html#gaaf8a051fb13cab1eba5e2149f75e902f>
