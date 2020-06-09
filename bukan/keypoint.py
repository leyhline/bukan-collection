"""
bukan.keypoint
~~~~~

High-level interface for bukan.py.

Takes one image as input (i.e. a file system path) and produces
a JSON file as output, representing a list of OpenCV KeyPoint objects.
<https://docs.opencv.org/4.2.0/d2/d29/classcv_1_1KeyPoint.html> 

:copyright: (c) 2020 Thomas Leyh <leyht@informatik.uni-freiburg.de>
:licence: GPLv3, see LICENSE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from bukan import detect_features