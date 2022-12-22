import os
from pickle import DICT
from typing import List

import numpy as np
from camera_toolkit.aruco import get_aruco_marker_poses
from camera_toolkit.utils import homogeneous_matrix
from camera_toolkit.zed2i import Zed2i, sl
import cv2
from scipy.spatial import transform
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

import enum


class CalibrationMode(enum.Enum):
    eye_in_hand = 1
    eye_to_hand = 2


def collect_eye_to_hand_transforms(marker_dict, marker_size_in_m: float, robot_ip="10.42.0.162"):
    """
    Collects two transforms (split into translation and rotation part):
        1. transform from robot to end-effector (i.e. the pose of the EE of the robot)
        2. transform from camera to calibration marker

    @param marker_dict:
    @param marker_size_in_m:
    @param robot_ip:
    @return:
    """
    eef_positions_in_robot_frame = []
    eef_rotmats_in_robot_frame = []
    marker_positions_in_camera_frame = []
    marker_rotmats_in_camera_frame = []

    zed = Zed2i(resolution=sl.RESOLUTION.HD2K, fps=15)
    camera_intrinsics = zed.get_camera_matrix()

    robot_control = RTDEControl(robot_ip)
    robot_receive = RTDEReceive(robot_ip)

    while True:

        print("Press Enter once the EEF and marker are in place. Press Q to stop")
        while True:
            robot_control.teachMode()
            img = zed.get_rgb_image()
            img = zed.image_shape_torch_to_opencv(img)
            img, t, r, ids = get_aruco_marker_poses(img, camera_intrinsics, marker_size_in_m, marker_dict, True)

            cv2.namedWindow("view", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("view", 800, 400)
            cv2.imshow("view", img)
            key = cv2.waitKey(100)

            if key == ord("q"):
                cv2.destroyAllWindows()
                robot_control.endTeachMode()
                return np.array(eef_positions_in_robot_frame), np.array(eef_rotmats_in_robot_frame), \
                       np.array(marker_positions_in_camera_frame), np.array(marker_rotmats_in_camera_frame)

            if key != -1:
                robot_control.endTeachMode()
                print("registering poses.")
                cv2.destroyAllWindows()
                break

        pose = robot_receive.getActualTCPPose()
        print(pose)
        eef__position_in_robot_frame = np.array(pose[:3])
        eef_rotmat_in_robot_frame = transform.Rotation.from_rotvec(np.array(pose[3:])).as_matrix()
        print(f" eef pose {eef__position_in_robot_frame}  - {eef_rotmat_in_robot_frame}")
        img = zed.get_rgb_image()
        img = zed.image_shape_torch_to_opencv(img)

        img, t, r, ids = get_aruco_marker_poses(img, camera_intrinsics, marker_size_in_m, marker_dict, True)
        print(f" marker in camera frame: {t} {r}")
        if t is None:
            continue

        eef_positions_in_robot_frame.append(eef__position_in_robot_frame)
        eef_rotmats_in_robot_frame.append(eef_rotmat_in_robot_frame)
        marker_positions_in_camera_frame.append(t[0])
        marker_rotmats_in_camera_frame.append(r[0])


def get_eye_to_hand_transforms(load_from: str = None, save_to_dir: str = None):
    """
    Record or fetch latest calibration data.
    @param save_to_dir: record new calibration data and persist to dir. Will save to currentdir if none.
    @param load_from: load existing calibration data. Will record new if none.
    @return: eef_pos_in_robot, eef_rotmat_in_robot, marker_pos_in_cam, marker_rot_matrix_in_camera
    """
    if load_from:
        eef_pos_in_robot = np.load(os.path.join(load_from, "eef_pos_in_robot.npy"))
        eef_rotmat_in_robot = np.load(os.path.join(load_from, "eef_rotmat_in_robot.npy"))
        marker_pos_in_cam = np.load(os.path.join(load_from, "marker_pos_in_cam.npy"))
        marker_rot_matrix_in_camera = np.load(os.path.join(load_from, "marker_rot_matrix_in_camera.npy"))

    else:
        # TODO: it should be clear, or documented, or configurable which aruco marker is being used
        aruco_marker_type = cv2.aruco.DICT_6X6_250
        marker_size_in_m = 0.106
        eef_pos_in_robot, eef_rotmat_in_robot, marker_pos_in_cam, marker_rot_matrix_in_camera = \
            collect_eye_to_hand_transforms(aruco_marker_type, marker_size_in_m)

        if not save_to_dir:
            save_to_dir = '.'

        np.save(os.path.join(save_to_dir, "eef_pos_in_robot.npy"), eef_pos_in_robot)
        np.save(os.path.join(save_to_dir, "eef_rotmat_in_robot.npy"), eef_rotmat_in_robot)
        np.save(os.path.join(save_to_dir, "marker_pos_in_cam.npy"), marker_pos_in_cam)
        np.save(os.path.join(save_to_dir, "marker_rot_matrix_in_camera.npy"), marker_rot_matrix_in_camera)

    return eef_pos_in_robot, eef_rotmat_in_robot, marker_pos_in_cam, marker_rot_matrix_in_camera


def do_eye_to_hand_calibration(save_to_dir: str = None):
    """"
     Eye to hand calibration uses opencv calibrateHandEye method formulation with different variable:
     Transform from gripper to base is inverted: transform from base to gripper.
     This is equivalent to the gripper_in_base coordinates (eef_pos_in_robot) to base_in_gripper coordinates (robot_pos_in_eef).
     See https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d

    """

    if not save_to_dir:
        save_to_dir = "."

    # 1. Collect calibration data
    eef_pos_in_robot, eef_rotmat_in_robot, marker_pos_in_cam, marker_rot_matrix_in_camera = get_eye_to_hand_transforms(
        save_to_dir=save_to_dir)

    # 2. invert translation and rotation component of Transform from gripper to base (here: eef_pos_in_robot to robot_pos_in_eef)
    # TODO: put this code behind utils method that inverts a transform (i.e. pos + rot) or a homog matrix. Note that it should be vectorized
    robot_rotmat_in_eef = np.linalg.inv(eef_rotmat_in_robot)  # also, might do transpose here, is == to inverse for rotation matrices
    robot_pos_in_eef = np.zeros_like(eef_pos_in_robot)
    for i in range(eef_pos_in_robot.shape[0]):
        robot_pos_in_eef[i, :] = (- robot_rotmat_in_eef[i, :, :] @ eef_pos_in_robot[i, :])

    camera_rotmat_in_base, camera_pos_in_base = cv2.calibrateHandEye(robot_rotmat_in_eef, robot_pos_in_eef,
                                                                     marker_rot_matrix_in_camera, marker_pos_in_cam)
    print(f" camera position = {camera_pos_in_base}")
    print(f" camera orientation in base =  \n {camera_rotmat_in_base}")

    camera_homog_transform_mat_in_base = homogeneous_matrix(camera_pos_in_base.flatten(), camera_rotmat_in_base)

    np.save(os.path.join(save_to_dir, "camera_rotmat_in_base.npy"), camera_rotmat_in_base)
    np.save(os.path.join(save_to_dir, "camera_pos_in_base.npy"), camera_pos_in_base)
    np.save(os.path.join(save_to_dir, "camera_homog_transform_mat_in_base.npy"), camera_homog_transform_mat_in_base)
    print(f'Saved calibration matrices in directory "{save_to_dir}"')


def do_eye_in_hand_calibration():
    eef_pos_in_robot, eef_rotmat_in_robot, marker_pos_in_cam, marker_rot_matrix_in_camera = get_eye_to_hand_transforms()

    camera_rotmat_in_gripper, camera_pos_in_gripper = cv2.calibrateHandEye(eef_rotmat_in_robot, eef_pos_in_robot,
                                                                           marker_rot_matrix_in_camera, marker_pos_in_cam)


if __name__ == "__main__":
    mode = CalibrationMode.eye_to_hand

    if mode == CalibrationMode.eye_in_hand:
        do_eye_in_hand_calibration()
    if mode == CalibrationMode.eye_to_hand:
        do_eye_to_hand_calibration()
