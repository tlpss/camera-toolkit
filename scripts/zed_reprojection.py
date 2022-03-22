"""
Script for testing the reprojection of an image coordinate
"""

from camera_toolkit.zed2i import Zed2i
from camera_toolkit.aruco_pose import get_aruco_marker_poses
import cv2
from camera_toolkit.reproject_to_z_plane import reproject_to_ground_plane

import numpy as np

Zed2i.list_camera_serial_numbers()
zed = Zed2i()
img = zed.get_mono_rgb_image()
img = zed.image_shape_torch_to_opencv(img)
cam_matrix = zed.get_mono_camera_matrix()
print(img.shape)

img, t, r = get_aruco_marker_poses(img, cam_matrix, 0.106, cv2.aruco.DICT_6X6_250, True)
print(t)
print(r)

transform = np.eye(4)
transform[:3, :3] = r[0]
transform[:3, 3] = t[0]
cv2.imshow(",", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_coords = []

x = int(input("x in image coords: "))
y = int(input("y in image coords: "))

image_coords.append([x, y])

image_coords = np.array(image_coords)
point = reproject_to_ground_plane(image_coords, cam_matrix, transform)
print(f" point in aruco base: {point}")


zed.close()
