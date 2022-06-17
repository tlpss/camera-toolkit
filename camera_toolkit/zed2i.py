import cv2
import numpy as np
import pyzed.sl as sl

from camera_toolkit.camera import BaseCamera


class Zed2i(BaseCamera):
    def __init__(self, resolution: sl.RESOLUTION = sl.RESOLUTION.HD720, fps=30) -> None:
        super().__init__()
        self.camera = sl.Camera()
        self.camera_params = sl.InitParameters()
        self.camera_params.camera_resolution = resolution
        self.camera_params.camera_fps = fps

        if self.camera.is_opened():
            # close to open with correct params
            self.camera.close()

        status = self.camera.open(self.camera_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise IndexError(f"could not open camera, error = {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.image_matrix = sl.Mat()  # allocate memory for view

    def close(self):
        self.camera.close()

    def get_mono_camera_matrix(self):
        fx = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.fx
        fy = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.fy
        cx = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.cx
        cy = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.cy
        print(fx)
        cam_matrix = np.zeros((3, 3))
        cam_matrix[0, 0] = fx
        cam_matrix[1, 1] = fy
        cam_matrix[2, 2] = 1
        cam_matrix[0, 2] = cx
        cam_matrix[1, 2] = cy
        return cam_matrix

    def get_mono_rgb_image(self) -> np.ndarray:
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


if __name__ == "__main__":
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i(sl.RESOLUTION.HD720)
    img = zed.get_mono_rgb_image()
    print(img.shape)
    img = zed.image_shape_torch_to_opencv(img)
    print(img.shape)
    cv2.imshow(",", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    zed.close()
