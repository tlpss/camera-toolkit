from typing import Tuple  # Math library

import cv2
import numpy as np  # Import Numpy library
from scipy.spatial.transform import Rotation as R


def get_aruco_marker_coords(
    frame: np.ndarray,
    aruco_dictionary_name: str,
):
    this_aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dictionary_name)
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()

    # Detect ArUco markers in the video frame
    (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
        frame, this_aruco_dictionary, parameters=this_aruco_parameters
    )
    # Check that at least one ArUco marker was detected
    if marker_ids is None:
        return None

    # average over the four corners of the markers to get the marker's center coordinates
    return np.mean(corners, axis = 2)


def get_aruco_marker_poses(
    frame: np.ndarray,
    cam_matrix: np.ndarray,
    aruco_marker_size: float,
    aruco_dictionary_name: str,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    this_aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dictionary_name)
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()

    # Detect ArUco markers in the video frame
    (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
        frame, this_aruco_dictionary, parameters=this_aruco_parameters
    )
    # Check that at least one ArUco marker was detected
    if marker_ids is None:
        return frame,None, None, None

        # Get the rotation and translation vectors
    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_size, cam_matrix, np.zeros(4))

    # The pose of the marker is with respect to the camera lens frame.
    # Imagine you are looking through the camera viewfinder,
    # the camera lens frame's:
    # x-axis points to the right
    # y-axis points straight down towards your toes
    # z-axis points straight ahead away from your eye, out of the camera
    translations = []
    rotation_matrices = []
    for i, marker_id in enumerate(marker_ids):
        # Store the rotation information
        rotation_matrix = np.eye(3)
        rotation_matrix = cv2.Rodrigues(np.array(rvecs[i][0]))[0]

        translations.append(tvecs[i][0])
        rotation_matrices.append(rotation_matrix)

    if visualize:
        frame = np.ascontiguousarray(frame)
        # Draw the axes on the marker
        for i in range(len(marker_ids)):
            cv2.drawFrameAxes(frame, cam_matrix, np.zeros(4), rvecs[i], tvecs[i], 0.05)
    return frame, translations, rotation_matrices, marker_ids


if __name__ == "__main__":
    from zed2i import Zed2i, sl

    """
    Test script with Zed2i
    """
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i(resolution=sl.RESOLUTION.HD2K, fps = 15)
    img = zed.get_mono_rgb_image()
    img = zed.image_shape_torch_to_opencv(img)
    cam_matrix = zed.get_mono_camera_matrix()
    print(img.shape)

    coords = get_aruco_marker_coords(img,cv2.aruco.DICT_5X5_250)
    print(coords)
    img, t, r, ids = get_aruco_marker_poses(img, cam_matrix, 0.058, cv2.aruco.DICT_5X5_250, True)
    print(t)
    print(r)
    cv2.imshow(",", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    zed.close()
