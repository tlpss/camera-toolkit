"""
Script for testing the reprojection of an image coordinate
"""

import cv2

from camera_toolkit.reproject import reproject_to_camera_frame
from camera_toolkit.zed2i import Zed2i, sl

Zed2i.list_camera_serial_numbers()
zed = Zed2i(sl.RESOLUTION.HD720)
img = zed.get_rgb_image()
img = zed.image_shape_torch_to_opencv(img)
cam_matrix = zed.get_camera_matrix()


cv2.imshow(",", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

x = int(input("x in image coords: "))
y = int(input("y in image coords: "))

point = reproject_to_camera_frame(x, y, cam_matrix, zed.get_depth_map())
print(point.shape)
print(f" point in camera frame: {point}")


zed.close()
