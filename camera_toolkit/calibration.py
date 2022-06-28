from pickle import DICT
import numpy as np
from camera_toolkit.aruco import get_aruco_marker_poses
from camera_toolkit.zed2i import Zed2i, sl
import cv2
from scipy.spatial import transform
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


def collect_eye_to_hand_data(marker_dict, marker_size: float, robot_ip="10.42.0.162"):
    eef_positions_in_robot_frame = []
    eef_rotmats_in_robot_frame = []
    marker_positions_in_camera_frame = []
    marker_rotmats_in_camera_frame = []

    zed = Zed2i(resolution=sl.RESOLUTION.HD2K, fps=15)
    cam_matrix = zed.get_camera_matrix()

    robot_control = RTDEControl(robot_ip)
    robot_receive = RTDEReceive(robot_ip)

    while (True):

        print("Press Enter once the EEF and marker are in place. Press Q to stop")
        while (True):
            robot_control.teachMode()
            img = zed.get_rgb_image()
            img = zed.image_shape_torch_to_opencv(img)
            img, t, r, ids = get_aruco_marker_poses(img, cam_matrix, marker_size, marker_dict, True)

            cv2.namedWindow("view", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("view", 800, 400)
            cv2.imshow("view", img)
            key = cv2.waitKey(100)
            if key == ord("q"):
                cv2.destroyAllWindows()
                robot_control.endTeachMode()
                return np.array(eef_positions_in_robot_frame), np.array(eef_rotmats_in_robot_frame), np.array(
                    marker_positions_in_camera_frame), np.array(marker_rotmats_in_camera_frame)
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

        img, t, r, ids = get_aruco_marker_poses(img, cam_matrix, marker_size, marker_dict, True)
        print(f" marker in camera frame: {t} {r}")
        if t is None:
            continue
        eef_positions_in_robot_frame.append(eef__position_in_robot_frame)
        eef_rotmats_in_robot_frame.append(eef_rotmat_in_robot_frame)
        marker_positions_in_camera_frame.append(t[0])
        marker_rotmats_in_camera_frame.append(r[0])


if __name__ == "__main__":

    mode = "eye-to-hand"
    assert mode in ("eye-in-hand", "eye-to-hand")
    #
    # eef_pos_in_robot, eef_rotmat_in_robot, marker_pos_in_cam, marker_rot_matrix_in_camera = collect_eye_to_hand_data(cv2.aruco.DICT_6X6_250,
    #                                                                                                                  0.103)
    #
    # np.save("eef_pos_in_robot.npy",eef_pos_in_robot)
    # np.save("eef_rotmat_in_robot",eef_rotmat_in_robot)
    # np.save("marker_pos_in_cam",marker_pos_in_cam)
    # np.save("marker_rot_matrix_in_camrea",marker_rot_matrix_in_camera)

    eef_pos_in_robot = np.load("eef_pos_in_robot.npy")
    eef_rotmat_in_robot = np.load("eef_rotmat_in_robot.npy")
    marker_pos_in_cam = np.load("marker_pos_in_cam.npy")
    marker_rot_matrix_in_camera = np.load("marker_rot_matrix_in_camrea.npy")

    if mode == "eye-in-hand":
        camera_rotmat_in_gripper, camera_pos_in_gripper = cv2.calibrateHandEye(eef_rotmat_in_robot, eef_pos_in_robot,
                                                                               marker_rot_matrix_in_camera, marker_pos_in_cam)
    if mode == "eye-to-hand":
        robot_rotmat_in_eef = np.linalg.inv(eef_rotmat_in_robot)
        assert np.allclose(robot_rotmat_in_eef[0], eef_rotmat_in_robot[0].T)
        robot_pos_in_eef = np.zeros_like(eef_pos_in_robot)
        for i in range(eef_pos_in_robot.shape[0]):
            robot_pos_in_eef[i, :] = (- robot_rotmat_in_eef[i, :, :] @ eef_pos_in_robot[i, :])

            camera_rotmat_in_base, camera_pos_in_base = cv2.calibrateHandEye(robot_rotmat_in_eef, robot_pos_in_eef, marker_rot_matrix_in_camera,
                                                                         marker_pos_in_cam)
        print(f" camera position = {camera_pos_in_base}")
        print(f" camera orientation in base =  \n {camera_rotmat_in_base}")

        np.save("camera_rotmat_in_base.npy", camera_rotmat_in_base)
        np.save("camera_pos_in_base.npy", camera_pos_in_base)

        camera_homog_transform_mat_in_base = np.zeros((4, 4))

        camera_homog_transform_mat_in_base[0:3, 0:3] = camera_rotmat_in_base
        camera_homog_transform_mat_in_base[0:3, 3] = camera_pos_in_base.flatten()
        camera_homog_transform_mat_in_base[3, :] = [0, 0, 0, 1]
        np.save("camera_homog_transform_mat_in_base.npy", camera_homog_transform_mat_in_base)
