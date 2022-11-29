import numpy as np


def homogeneous_vector(points: np.ndarray):
    """
    Args:
        points: (3, N) array of points, each column is a set of coordinates
    """
    points_homog = np.vstack((points, np.ones(points.shape[1])))
    return points_homog


def homogeneous_matrix(translation_vector: np.ndarray, rotation_matrix: np.ndarray):
    translation_vector = translation_vector.reshape((3,))
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix should be shape (3,3) and not {rotation_matrix.shape}")

    hommat = np.zeros((4, 4))
    hommat[0:3, 0:3] = rotation_matrix
    hommat[0:3, 3] = translation_vector
    hommat[3, :] = [0, 0, 0, 1]

    return hommat
