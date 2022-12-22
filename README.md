# camera-toolkit

python package for working with RGB(D) cameras in the context of robotic manipulation.

And has following features:

- convenient access to the camera output by wrapping the API (currently only supports the ZED2i Camera)
- camera-agnostic code for detecting fiducial markers ( for extrinsics calibration or object localisation)
- camera-agnostic code for (re)projection of points back to the camera frame or the world frame, using prior knowledge
  on the objects; multi-view (TODO) or depth maps.

## ZED 2i

Using the Zed cameras requires installation of the SDK and the python wrapper.

See https://www.stereolabs.com/docs/app-development/python/install/#getting-started for more info.

If you already have the SDK, run the get_python_api script. If you don't have the SDK already, install the SDK which
will also install the python API (if you choose to).

## Naming

###

eef_pos_in_robot = position of the end-effector in the frame of the robot base joint.
This corresponds to the transform FROM robot base TO end-effector.

marker_pos_in_cam = position of the aruco calibration pattern in the frame of the camera.
This corresponds to the transform FROM the camera TO the calibration pattern.

### Abbreviations

| **Abbreviation** | **Meaning**       |
|------------------|-------------------|
| eef              | End-effector      |
| rotmat           | Rotation matrix   |
| pos              | Position          |
| rot              | Rotation          |
| hommat           | Homogeneous matrix | 

