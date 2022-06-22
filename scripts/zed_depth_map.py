"""
script to query the depth map on a pixel in the rgb image for testing depth settings or validating the distances
"""

import cv2

from camera_toolkit.reproject import extract_depth_from_depthmap_heuristic
from camera_toolkit.zed2i import Zed2i

if __name__ == "__main__":
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()
    img = zed.get_rgb_image()
    print(img.shape)
    img = zed.image_shape_torch_to_opencv(img)
    print(img.shape)
    depth_map = zed.get_depth_map()
    depth_image = zed.get_dept_image()

    def mouse_callback(event, x, y, flags, params):
        if event == 2:
            print(f"img coords {x, y} -> z = {depth_map[y,x]} , z heuristic = {extract_depth_from_depthmap_heuristic(x,y,depth_map)}")

    cv2.namedWindow("test")
    cv2.setMouseCallback("test", mouse_callback)
    cv2.imshow("test", img)
    cv2.imshow("depth_image",depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    zed.close()
