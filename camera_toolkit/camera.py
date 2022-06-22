import abc

import numpy as np


class BaseCamera(abc.ABC):
    @abc.abstractmethod
    def get_camera_matrix():
        pass
    @abc.abstractmethod
    def get_rgb_image():
        pass

    @abc.abstractmethod
    def get_depth_map():
        pass

    @staticmethod
    def image_shape_opencv_to_torch(image: np.ndarray) -> np.ndarray:
        #  h, x w, c -> channels x h x w
        image = np.moveaxis(image, -1, 0)
        # BGR -> RGB
        image = image[[2, 1, 0], :, :]
        return image

    @staticmethod
    def image_shape_torch_to_opencv(image: np.ndarray) -> np.ndarray:
        # RGB -> BGR
        image = image[[2, 1, 0], :, :]
        # cxhx w -> h x w x c
        image = np.moveaxis(image, 0, -1)
        return image
