import os
import pathlib

import cv2

from camera_toolkit.zed2i import Zed2i

zed = Zed2i()


index = 0
path = pathlib.Path(__file__).parent.resolve() / "data"
assert os.path.exists(path)

while True:

    img = zed.get_mono_rgb_image()
    img = zed.image_shape_torch_to_opencv(img)
    # Show images
    cv2.imshow("image", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
        print(f"image {index} saved")
        cv2.imwrite(f"{path}/img{index}.png", img)
        index += 1
