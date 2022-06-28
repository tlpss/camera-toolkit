import os
import cv2
from cv2 import aruco
import numpy as np
from PIL import Image

# Create aruco marker

# Create aruco board

# Create chessboard aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
nr_markers_x_direction = 5
nr_markers_y_direction = 7
length_of_marker_side = 0.04  # in meter.
length_of_marker_separation = 0.01
board = aruco.GridBoard_create(nr_markers_x_direction, nr_markers_y_direction,
                               length_of_marker_side, length_of_marker_separation, aruco_dict)

A4 = (3508, 2408)
A3 = (4961, 3508)
img_resolution = A3
img_raw = board.draw(img_resolution)
img = Image.fromarray(img_raw)
img.save('aruco.png')
img.show()


arucoParams = aruco.DetectorParameters_create()
