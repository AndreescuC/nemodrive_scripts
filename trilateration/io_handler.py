import yaml
import numpy as np
from datetime import datetime


def read_config():

    with open("config.yaml", 'r') as stream:

        try:
            conf = yaml.load(stream)

            intrinsics = conf['intrinsics']

            base_world_coordinates = {k: np.array(v) for k, v in conf['windshield_relative_positions'].items()}

            session_info = "Extrinsic calibration, %s, session %s, camera[pid:%d][sid:%d](%s)" \
                           % (conf['day'], conf['session'], int(conf['camera']['pid']), int(conf['camera']['sid']), conf['camera']['position'])

            distances = [
                {
                    windshield_point: float(distance) / 100 #cm to m
                    for windshield_point, distance in zip(base_world_coordinates.keys(), landmark[2:6])
                }
                for landmark in conf['landmarks']
            ]
            camera_points = [
                (landmark[0], landmark[1])
                for landmark in conf['landmarks']
            ]
            known_zs = [
                landmark[6]
                for landmark in conf['landmarks']
            ]
        except yaml.YAMLError as exc:
            print(exc)
            return None, None, None, None

    return session_info, intrinsics, distances, base_world_coordinates, camera_points, known_zs


def log_results(session_info, tvec, rvec, errors):
    with open("result.log", 'a') as f:
        f.write(
            "%s:\n\t tvec: [%f %f %f],\n\t rvec: [%f %f %f]\n\tWith errors: %s\n\t Recorded at %s\n\n" %
            (session_info, tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2], errors.__repr__(), datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        )
