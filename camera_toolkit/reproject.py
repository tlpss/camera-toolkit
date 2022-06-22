import numpy as np 

def reproject_image_coord(x: int, y: int, camera_matrix: np.ndarray, depth_map: np.ndarray):
    z = 