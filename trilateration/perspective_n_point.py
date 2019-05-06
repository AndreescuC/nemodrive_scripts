import cv2
import numpy as np


def solve_pnp(camera_points, world_points, intrinsics):
    camera_matrix = np.array(intrinsics['camera_matrix'])
    distortion_coefs = np.array(intrinsics['distortion_coefs'])
    world_points = np.array(world_points, dtype='Float64')
    camera_points = np.array(camera_points, dtype='Float64')

    ret, rvec, tvec = cv2.solvePnP(world_points, camera_points, camera_matrix, distortion_coefs)

    if not ret:
        print("Error at solve pnp")
        return None, None

    return tvec, rvec
