import numpy as np


class ExtrinsicModel:

    def __init__(self, target_camera_tvec, target_camera_rvec, reference_camera_tvec, reference_camera_rvec):
        self.target_camera_tvec = target_camera_tvec
        self.target_camera_rvec = target_camera_rvec
        self.reference_camera_tvec = reference_camera_tvec
        self.reference_camera_rvec = reference_camera_rvec

    def train(self):
        print("pass")

    def transform(self, point):
        assert len(point) == 2


    def predict_points(self, points):
        return np.array([self.transform(point) for point in points])
