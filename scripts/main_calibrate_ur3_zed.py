from camera_toolkit.calibration import do_eye_to_hand_calibration


def do_calibration():
    save_dir = '.'
    do_eye_to_hand_calibration(save_dir)
    pass


if __name__ == '__main__':
    do_calibration()