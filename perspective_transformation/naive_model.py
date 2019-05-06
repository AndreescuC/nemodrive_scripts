import numpy as np


class IdentityModel:

    def get_name(self):
        return "IdentityModel"

    def train(self):
        print("pass")

    def predict_points(self, points):
        return np.array(points)


class NaiveTranslationModel:

    def __init__(self,  delta_y=0, delta_x=0):
        self.delta_y = delta_y
        self.delta_x = delta_x

    def get_name(self):
        return "NaiveTranslationModel(x=%d y=%d)" % (self.delta_x, self.delta_y)

    def train(self):
        print("pass")

    def predict_points(self, points):
        return np.array([[point[0] + self.delta_y, point[1] + self.delta_x] for point in points])

