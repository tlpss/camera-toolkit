import cv2
import numpy as np
import pyzed.sl as sl

from camera_toolkit.camera import BaseCamera


class Zed2i(BaseCamera):
    def __init__(self, resolution: sl.RESOLUTION = sl.RESOLUTION.HD2K, fps=15, serial_number=35357320) -> None:
        # TODO: make depth settings configurable.
        # TODO: make camera ID configurable.

        super().__init__()
        self.camera = sl.Camera()
        self.camera_params = sl.InitParameters()
        self.camera_params.camera_resolution = resolution
        self.camera_params.camera_fps = fps
        self.camera_params.set_from_serial_number(serial_number)

        # https://www.stereolabs.com/docs/depth-sensing/depth-settings/
        self.camera_params.depth_mode = sl.DEPTH_MODE.NEURAL  # the Neural mode gives far better results usually
        self.camera_params.coordinate_units = sl.UNIT.METER
        self.camera_params.depth_minimum_distance = (
            0.3  # objects closerby will have artifacts so they are filtered out (querying them will give a - Infinty)
        )
        self.camera_params.depth_maximum_distance = 2.0  # filter out far away objects

        if self.camera.is_opened():
            # close to open with correct params
            self.camera.close()

        status = self.camera.open(self.camera_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise IndexError(f"could not open camera, error = {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD  # standard > fill for accuracy. See docs.

        print(self.runtime_params)
        self.image_matrix = sl.Mat()  # allocate memory for RGB view
        self.depth_matrix = sl.Mat()  # allocate memory for the depth map

    def close(self):
        self.camera.close()

    def get_camera_matrix(self, camera: str = "left") -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: 3x3 matrix containing
            [ fx  0  cx
              0   fy cy
              0   0  1 ]

            where fx, fy are the focal distances expressed in pixels and cx,cy the offset of the image coordinate system (topleft) to the principal point
        """
        assert camera in ("left", "right")
        if camera == "left":
            fx = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
            fy = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
            cx = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
            cy = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
        else:
            fx = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.fx
            fy = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.fy
            cx = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.cx
            cy = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.cy
        cam_matrix = np.zeros((3, 3))
        cam_matrix[0, 0] = fx
        cam_matrix[1, 1] = fy
        cam_matrix[2, 2] = 1
        cam_matrix[0, 2] = cx
        cam_matrix[1, 2] = cy
        return cam_matrix

    def get_rgb_image(self, camera="left") -> np.ndarray:

        assert camera in ("left", "right")

        error_code = self.camera.grab(self.runtime_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise IndexError("Could not grab new camera frame")
        if camera == "right":
            view = sl.VIEW.RIGHT
        else:
            view = sl.VIEW.LEFT
        self.camera.retrieve_image(self.image_matrix, view)
        image = self.image_matrix.get_data()
        return self.image_shape_opencv_to_torch(image)

    def get_dept_image(self) -> np.ndarray:
        """
        Returns an 8 bit quantization of the depth map. Should only be used for visualization.
        """
        error_code = self.camera.grab(self.runtime_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise IndexError("Could not grab new camera frame")
        self.camera.retrieve_image(self.image_matrix, sl.VIEW.DEPTH)
        image = self.image_matrix.get_data()
        return image

    def get_depth_map(self) -> np.ndarray:
        error_code = self.camera.grab(self.runtime_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise IndexError("Could not grab new camera frame")
        self.camera.retrieve_measure(self.depth_matrix, sl.MEASURE.DEPTH)
        depth_map = self.depth_matrix.get_data()
        return depth_map

    @staticmethod
    def list_camera_serial_numbers():
        """
        List all connected ZED cameras

        can be used to select a device ID or to check if cameras are connected.
        """
        device_list = sl.Camera.get_device_list()
        print(device_list)
        return device_list


if __name__ == "__main__":
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i(sl.RESOLUTION.HD2K)
    img = zed.get_rgb_image()
    print(img.shape)
    img = zed.image_shape_torch_to_opencv(img)
    print(img.shape)
    depth_map = zed.get_depth_map()
    depth_image = zed.get_dept_image()

    cv2.imshow("test", img)
    cv2.imshow(",", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    zed.close()
