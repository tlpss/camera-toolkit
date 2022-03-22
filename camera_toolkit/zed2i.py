from camera_toolkit.camera import BaseCamera
import numpy as np 
import pyzed.sl as sl
import cv2
from enum import Enum

class RGBSensor(Enum):
    LEFT = sl.VIEW.LEFT
    RIGHT = sl.VIEW.RIGHT

class Zed2i(BaseCamera):

    def __init__(self, ) -> None:
        super().__init__()
        self.camera = sl.Camera()
        self.camera_params = sl.InitParameters()
        self.camera_params.camera_resolution = sl.RESOLUTION.HD720
        self.camera_params.camera_fps = 30

        if self.camera.is_opened():
            self.camera.close() 
            # close to open with correct params 
        status = self.camera.open(self.camera_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise IndexError(f"could not open camera, error = {status}")
        
        self.runtime_params = sl.RuntimeParameters()
        self.image_matrix = sl.Mat() # allocate memory for view

    def close(self):
        self.camera.close()

    def get_mono_camera_matrix(self):
        fx = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
        fy = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
        cx = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
        cy = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
        print(fx)
        cam_matrix = np.zeros((3,3))
        cam_matrix[0,0] = fx
        cam_matrix[1,1] = fy
        cam_matrix[2,2] = 1
        cam_matrix[0,2] = cx
        cam_matrix[1,2] = cy
        return cam_matrix

    def get_mono_rgb_image(self, camera = RGBSensor.LEFT) -> np.ndarray:
        error_code = self.camera.grab(self.runtime_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise IndexError("Could not grab new camera frame")
        self.camera.retrieve_image(self.image_matrix, sl.VIEW.RIGHT)
        image = self.image_matrix.get_data()
        return self.image_shape_opencv_to_torch(image)

    @staticmethod
    def list_camera_serial_numbers():
        device_list = sl.Camera.get_device_list()
        print(device_list)
        return device_list


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
    
if __name__ == "__main__":
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()
    img= zed.get_mono_rgb_image()
    print(img.shape)
    img = zed.image_shape_torch_to_opencv(img)
    print(img.shape)
    cv2.imshow(",",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    zed.close()
