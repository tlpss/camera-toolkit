import numpy as np


def reproject_to_world_z_plane(
    image_coords: np.ndarray, camera_matrix: np.ndarray, world_in_camera_frame_pose: np.ndarray, height: float = 0.0
):
    """Reprojects points from the camera plane to the specified Z-plane of the world frame
    (as this is often the frame in which you have the z-information)

    This is useful if you known the height of the object in the world frame,
     which is the case for 2D items (cloth!) or for rigid, known 3D objects (that do not tumble)

    Args:
        image_coords (_type_): 2D (Nx2) numpy vector with (u,v) coordinates of a point w.r.t. the camera matrix
        camera_matrix (_type_): 2D 3x3 numpy array with camera matrix
        world_in_camera_frame_pose (_type_): 2D 4x4 numpy array with the transformation in homogeneous coordinates from the camera to the world origin.

    Returns:
        _type_ Nx3 numpy axis with world coordinates on the Z=height plane wrt to the world frame.
    """
    if image_coords.shape[0] == 0:
        return []

    coords = np.ones((image_coords.shape[0], 3))
    coords[:, :2] = image_coords
    image_coords = np.transpose(coords)

    camera_in_world_frame_pose = np.linalg.inv(world_in_camera_frame_pose)

    camera_frame_ray_vector = np.linalg.inv(camera_matrix) @ image_coords

    translation = camera_in_world_frame_pose[0:3, 3]
    rotation_matrix = camera_in_world_frame_pose[0:3, 0:3]

    world_frame_ray_vectors = rotation_matrix @ camera_frame_ray_vector
    world_frame_ray_vectors = np.transpose(world_frame_ray_vectors)
    t = (height - translation[2]) / world_frame_ray_vectors[:, 2]
    points = t[:, np.newaxis] * world_frame_ray_vectors + translation
    return points


def reproject_to_camera_frame(u: int, v: int, camera_matrix: np.ndarray, depth_map: np.ndarray, mask_size=11, depth_percentile=0.05):
    """
    reprojects a point on the image plane to the 3D frame of the camera.
    point = (u,v , 0) with origin in the top left corner of the img and y-axis point down

    Args:
        u:
        v:
        camera_matrix: 3x3 camera matrix
        depth_map: MxN depth map, depth_map at coord (u,v) gives the z-value of the position of that pixel in the camera frame (!not the distance to the camera!)

    Returns: (3,) np.array containing the coordinates of the point in the camera frame.

    """
    img_coords = np.array([u, v, 1.0])
    ray_in_camera_frame = np.linalg.inv(camera_matrix) @ img_coords  # shape is casted by numpy to column vector!

    z_in_camera_frame = extract_depth_from_depthmap_heuristic(u,v,depth_map,mask_size,depth_percentile)
    t = z_in_camera_frame / ray_in_camera_frame[2]

    position_in_camera_frame = t * ray_in_camera_frame
    return position_in_camera_frame


def extract_depth_from_depthmap_heuristic(
    u: int, v: int, depth_map: np.ndarray, mask_size: int = 11, depth_percentile: float = 0.05
) -> float:
    """
    A simple heuristic to get more robust depth values of the depth map. Especially with keypoints we are often interested in points
    on the edge of an object, or even worse on a corner. Not only are these regions noisy by themselves but the keypoints could also be
    be a little off.

    This function takes the percentile of a region around the specified point and assumes we are interested in the nearest object present.
    This is not always true (think about the backside of a box looking under a 45 degree angle) but it serves as a good proxy. The more confident
    you are of your keypoints and the better the heatmaps are, the lower you could set the mask size and percentile. If you are very, very confident
    you could directly take the pointcloud as well instead of manually querying the heatmap, but I find that they are more noisy.

    Also note that this function assumes there are no negative infinity values (no objects closer than 30cm!)
    """
    assert mask_size % 2, "only odd sized markers allowed"
    assert (
        depth_percentile < 0.25
    ), "For straight corners, about 75 percent of the region will be background.. Are your sure you want the percentile to be lower?"
    depth_region = depth_map[v - mask_size // 2 : v + mask_size // 2, u - mask_size // 2 : u + mask_size // 2]
    depth = np.nanquantile(depth_region.flatten(), depth_percentile)
    return depth


def project_world_to_image_plane(point: np.ndarray, world_to_camera_transform: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Projects a point from the 3D world frame to the 2D image plane.
    
    Works in two steps. First transforms the 3D point to a 3D point in camera frame. 
    Then projects the point onto the image plane. Note the normalization by the third coordinate."""
    point = np.array(point).reshape((3,1))
    point_homogeneous = np.append(point, [[1.0]])
    point_camera_homogeneous = world_to_camera_transform @ point_homogeneous
    point_camera = point_camera_homogeneous[:3]
    point_image_homogeneous = camera_matrix @ point_camera
    point_image = point_image_homogeneous[:2] / point_image_homogeneous[2]
    return point_image

