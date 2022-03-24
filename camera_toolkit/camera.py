import abc
import numpy as np

class BaseCamera(abc.ABC):

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def get_mono_rgb_image(self) -> np.ndarray:
        """ 

        Returns:
            np.ndarray: a 3 x W x H matrix containing the RGB channels (Pytorch format format)
        """
        pass

    @abc.abstractmethod
    def get_mono_camera_matrix(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: 3x3 matrix containing 
            [ fx  0  cx 
              0   fy cy
              0   0  1 ]
            
            where fx, fy are the focal distances expressed in pixels  and cx,cy the offset of the image coordinate system (topleft) to the principal point
        """
        pass

    @staticmethod
    def image_shape_opencv_to_torch(image: np.ndarray) -> np.ndarray:
        #  h, x w, c -> channels x h x w
        image = np.moveaxis(image, -1, 0)
        # BGR -> RGB
        image = image[[2,1,0],:,:]
        return image

    @staticmethod
    def image_shape_torch_to_opencv(image:np.ndarray) -> np.ndarray:
        # RGB -> BGR
        image = image[[2,1,0],:,:]
        # cxhx w -> h x w x c
        image = np.moveaxis(image, 0, -1)
        return image
    
