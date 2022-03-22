import abc
import numpy as np

class BaseCamera(abc.ABC):

    def __init__(self) -> None:
        super().__init__()

    def get_mono_rgb_image(self) -> np.ndarray:
        """ 

        Returns:
            np.ndarray: a W x H x 3 matrix containing the BGR(!) channels
        """
        pass

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


    
