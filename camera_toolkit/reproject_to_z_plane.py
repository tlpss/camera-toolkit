import numpy as np


def reproject_to_ground_plane(
    image_coords: np.ndarray, camera_matrix: np.ndarray, world_in_camera_frame_pose: np.ndarray, height: float = 0.0):
    """Reprojects points from the camera plane to the Z-plane of the world frame.

    Args:
        image_coords (_type_): 2D (Nx2) numpy vector with (u,v) coordinates of a point w.r.t. the camera matrix
        camera_matrix (_type_): 2D 3x3 numpy array with camera matrix
        world_in_camera_frame_pose (_type_): 2D 4x4 numpy array with the transformation in homogeneous coordinates from the camera to the world origin.

    Returns:
        _type_ Nx3 numpy axis with world coordinates on the Z=height plane wrt to the world frame.
    """
    coords = np.ones((image_coords.shape[0], 3))
    coords[:, :2] = image_coords
    image_coords = np.transpose(coords)

    camera_in_world_frame_pose = np.linalg.inv(world_in_camera_frame_pose)

    camera_frame_ray_vector = np.linalg.inv(camera_matrix) @ image_coords

    translation = camera_in_world_frame_pose[0:3, 3]
    rotation_matrix = camera_in_world_frame_pose[0:3, 0:3]

    world_frame_ray_vectors = rotation_matrix @ camera_frame_ray_vector
    world_frame_ray_vectors = np.transpose(world_frame_ray_vectors)
    t =  (height- translation[2])  /world_frame_ray_vectors[:, 2] 
    points = t[:, np.newaxis] * world_frame_ray_vectors + translation
    return points


if __name__ == "__main__":
    camera_matrix = np.array([[523.8, 0, 640.03], [0, 523.8, 367.03], [0, 0, 1]])
    transform = np.array(
        [
            [0.9956, 0.0192, 0.0915, 0.1656],
            [0.0423, -0.965, -0.257, 0.196],
            [0.083, 0.260, -0.962, 0.966],
            [0, 0, 0, 1],
        ]
    )
    image_coords = np.array([[578, 448], [300, 400]])

    print(reproject_to_ground_plane(image_coords, camera_matrix, transform))

    # [[-0.28038469  0.03821266  0.        ]
    # [-0.77474858  0.10849364  0.        ]]
