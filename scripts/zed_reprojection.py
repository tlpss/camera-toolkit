"""
Script for testing the reprojection of an image coordinate
"""

import cv2
import numpy as np

from camera_toolkit.reproject import reproject_to_camera_frame
from camera_toolkit.zed2i import Zed2i, sl

# Init camera
Zed2i.list_camera_serial_numbers()
zed = Zed2i(sl.RESOLUTION.HD2K)
camera_intrinsics_matrix = zed.get_camera_matrix()
camera_extrinsics_hommatrix = np.load("/home/adverley/code/projects/ur3tools/camera-toolkit/camera_toolkit/camera_homog_transform_mat_in_base.npy")


# capture mouse location click events
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        point = reproject_to_camera_frame(x, y, camera_intrinsics_matrix, zed.get_depth_map())
        print(point.shape)
        print(f" point in camera frame: {point}")

        point_homog = np.zeros((4,1))
        point_homog[:3] = point.reshape((3,1))
        point_homog[3] = 1
        point_in_base_frame = camera_extrinsics_hommatrix @ point_homog
        print(f'Point in base frame: {point_in_base_frame}')

# make screens
window_name = "Reprojection test"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click)

# keep on showing zed images
key = ''
while key != 113:  # for 'q' key
    img = zed.get_rgb_image()
    img = zed.image_shape_torch_to_opencv(img)
    cv2.imshow(window_name, img)
    cv2.waitKey(10)


cv2.destroyAllWindows()

zed.close()
