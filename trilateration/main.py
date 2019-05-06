import numpy as np
import io_handler as io
import trilateration
import perspective_n_point as pnp


def covert_to_clockwise_axe_system(world_points):
    return [[point[0], -point[2], point[1]] for point in world_points]


def convert_coord_convention(stations_coord):
    return {k: np.array([station[0], -station[2], station[1]]) for k, station in stations_coord.items()}


def compute_single_sphere_error(station, distance, world_coord):
    return pow(world_coord[0] - station[0], 2) +\
           pow(world_coord[1] - station[1], 2) +\
           pow(world_coord[2] - station[2], 2) -\
           pow(distance, 2)


def compute_errors(world_points, stations, distances):
    errors = []
    for landmark_dist, landmark_location in zip(distances, world_points):
        current_landmark_errors = [
            compute_single_sphere_error(stations[station_name], landmark_dist[station_name], landmark_location)
            for station_name in landmark_dist.keys()
        ]
        errors.append(current_landmark_errors)

    return errors


def main():
    session_info, intrinsics, distances, base_world_coordinates, camera_points, known_zs = io.read_config()
    base_world_coordinates = convert_coord_convention(base_world_coordinates)
    world_points = trilateration.solve_range_multilateration(base_world_coordinates, distances, known_zs=known_zs)

    errors = compute_errors(world_points, base_world_coordinates, distances)
    print("Landmarks world coordinates errors: ", errors)

    world_points = covert_to_clockwise_axe_system(world_points)
    tvec, rvec = pnp.solve_pnp(camera_points, world_points, intrinsics)
    io.log_results(session_info, tvec, rvec, errors)


if __name__ == '__main__':
    main()
