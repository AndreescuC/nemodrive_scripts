import yaml
import datetime
import numpy as np
from math import radians
import cv2
from argparse import Namespace
from car_utils import get_car_path, get_radius, get_car_line_mark, WHEEL_STEER_RATIO

FILTER_NEGATIVE_POINTS = True
FILTER_NONMONOTONIC_POINTS = True

CAR_L = 2.634  # Wheel base
CAR_T = 1.733  # Tread
MIN_TURNING_RADIUS = 5.
MAX_STEER = 500
SUPPORT_LINES = True


#TRACKBAR_GRANULARITY_FACTOR = 1
#TRACKBAR_GRANULARITY_FACTOR = 2
TRACKBAR_GRANULARITY_FACTOR = 4

MAX_TVEC_VAR: 0.1


class TurnRadius:
    def __init__(self, cfg):
        self.car_l = cfg.car_l
        self.car_t = cfg.car_t
        self.min_turning_radius = cfg.min_turning_radius

        self.num_points = num_points = 400
        max_wheel_angle = np.rad2deg(np.arctan(CAR_L / MIN_TURNING_RADIUS))
        self.angles = np.linspace(-max_wheel_angle, max_wheel_angle, num_points)

    def get_car_path(self, steer_factor, distance=20):
        """
        :param steer_factor: [-1, 1] (max left max right)
        :return:
        """
        num_points = self.num_points
        idx = np.clip(int(num_points/2. * steer_factor + num_points/2.), 0, num_points-1)
        r = get_radius(self.angles[idx])
        c, lw, rw = get_car_path(r, distance=distance)
        return c, lw, rw


class TrajectoryVisualizer:
    """
        Class that takes input a configuration file that contain information about line color and
        width, parameters about the camera and other data about the distance marks.
        Provides a method that projects the trajectory of the car given a steering angle on an
        image based on parameters from configuration.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.rvec = np.array(cfg.rvec)
        self.center_color = cfg.center_color
        self.center_width = cfg.center_width
        self.side_color = cfg.side_color
        self.side_width = cfg.side_width
        self.curve_length = cfg.curve_length
        self.initial_steer = cfg.initial_steer
        self.tvec = np.array(cfg.tvec)
        self.tvec_cpy = np.array(cfg.tvec)
        self.camera_matrix = np.array(cfg.camera_matrix)
        self.distortion = np.array(cfg.distortion)
        self.mark_count = cfg.mark_count
        self.start_dist = cfg.start_dist
        self.gap_dist = cfg.gap_dist
        self.distance_mark = cfg.distance_mark
        self.distance_mark_cpy = cfg.distance_mark
        print("Distance: %f" % self.distance_mark)
        #self.max_rvec_value = cfg.max_rvec_value

    def update_rvec(self, value, index):
        self.rvec[index] = value

    def update_tvec(self, value, index):
        self.tvec[index] = self.tvec_cpy[index] + value
        print("new value %f" % self.tvec[index])

    def update_distance(self, value):
        self.distance_mark = self.distance_mark_cpy + value
        print("new value %f" % self.distance_mark)

    def project_points_on_image_space(self, points_3d):
        rvec = self.rvec
        tvec = self.tvec
        camera_matrix = self.camera_matrix
        distortion = self.distortion

        points = np.array(points_3d).astype(np.float32)
        points, _ = cv2.projectPoints(points, rvec, np.array([0., 0., 0.]), camera_matrix, distortion)
        points = points.astype(np.int)
        points = points.reshape(-1, 2)
        return points

    def render_line(self, image, points, color=None, thickness=None,
                    filter_negative=FILTER_NEGATIVE_POINTS,
                    filter_nonmonotonic=FILTER_NONMONOTONIC_POINTS):
        points = self.project_points_on_image_space(points)

        h, w, _ = image.shape

        if filter_negative:
            idx = 0
            while (points[idx] < 0).any() or points[idx+1][1] < points[idx][1]:
                idx += 1
                if idx >= len(points) - 1:
                    break

            points = points[idx:]
        else:
            points = points[(points >= 0).all(axis=1)]

        # TODO Check validity - expect monotonic modification on x
        # monotonic decrease on y
        if len(points) <= 0:
            return image

        prev_x, prev_y = points[0]
        idx = 1

        if len(points) > 1 and filter_nonmonotonic:
            while points[idx][1] < prev_y:
                prev_x, prev_y = points[idx]
                idx += 1
                if idx >= len(points):
                    break

            valid_points = []
            while points[idx][0] >= 0 and points[idx][0] < w and idx < len(points)-1:
                valid_points.append([points[idx][0], points[idx][1]])
                idx += 1
        else:
            valid_points = [[prev_x, prev_y]]

        if filter_nonmonotonic:
            points = np.array(valid_points)

        points[:, 0] = np.clip(points[:, 0], 0, w)
        points[:, 1] = np.clip(points[:, 1], 0, h)
        for p in zip(points, points[1:]):
            image = cv2.line(image, tuple(p[0]), tuple(p[1]), color=color, thickness=thickness)

        return image

    def detect_indexes_on_lane_points_for_distance_marks(self, mark_count, start_dist, dist_gap):
        initial_steer = self.initial_steer
        curve_length = self.curve_length

        c, lw, rw = get_car_path(initial_steer, distance=curve_length)
        lw = self.add_3rd_dim(lw)

        def create_horizontal_line_at_depth(distance_from_camera, left_limit=-CAR_T/2, right_limit=CAR_T/2, n=2):
            x = np.expand_dims(np.linspace(left_limit, right_limit, num=n), axis=1)
            y = np.ones((n, 1))  # * CAMERA_HEIGHT - TODO some hardcoded value
            z = np.ones((n, 1)) * distance_from_camera
            xy = np.concatenate((x, y), axis=1)
            xyz = np.concatenate((xy, z), axis=1)
            return xyz

        def get_idx_closest_point_to(points, point):
            dists = list(map(lambda x : np.linalg.norm(x - point), points))
            min_idx = dists.index(min(dists))
            return min_idx

        indexes = []
        for dist in np.arange(start_dist, start_dist + mark_count * dist_gap, dist_gap):
            line_at_dist = create_horizontal_line_at_depth(dist)
            indexes.append(get_idx_closest_point_to(lw, line_at_dist[0]))

        return indexes

    def add_3rd_dim(self, points, height_provided=False):
        points3d = []
        if not height_provided:
            for point in points:
                points3d.append([
                    point[0] - self.tvec[0],
                    0 + self.tvec[1],
                    point[1] + self.tvec[2]]
                )
        else:
            for point in points:
                points3d.append([
                    point[0] - self.tvec[0],
                    point[1] + self.tvec[1],
                    point[2] + self.tvec[2]]
                )

        return np.array(points3d)

    def render_steer(self, image, steer_angle):
        r = get_radius(steer_angle / WHEEL_STEER_RATIO)
        return self.render(image, r)

    def render(self, image, radius):
        mark_count = self.mark_count
        start_dist = self.start_dist
        gap_dist = self.gap_dist
        curve_length = self.curve_length
        center_color = self.center_color
        center_width = self.center_width
        side_color = self.side_color
        side_width = self.side_width

        indexes = self.detect_indexes_on_lane_points_for_distance_marks(mark_count,
                                                                        start_dist,
                                                                        gap_dist)

        c, lw, rw = get_car_path(radius, distance=curve_length)
        # print(lw[2:])
        c = self.add_3rd_dim(c)
        lw = self.add_3rd_dim(lw)
        rw = self.add_3rd_dim(rw)

        if SUPPORT_LINES:
            x_factor = 1 #stanga dreapta
            y_factor = -1# sus jos
            z_factor = 1 #fata spate

            wheel_point_y = z_factor * (2.634 + 3)
            wheel_point_X = x_factor * (0.50874 + 0.37776 + 0.)

            corner_y = z_factor * self.distance_mark
            corner_x = wheel_point_X

            end_y = corner_y
            end_x = -x_factor * 20
            corner_vertical_z = y_factor * 10

            center_car_point_y = z_factor * (2.634 + 3)
            center_car_point_x = 0
            center_border_y = z_factor * self.distance_mark
            center_border_x = 0

            support_lw = [center_car_point_x, center_car_point_y]
            support_rw = [center_border_x, center_border_y]
            line = np.array([support_lw, support_rw])
            line = self.add_3rd_dim(line)
            # print("center line", line)
            image = self.render_line(image, line, color=(0, 0, 255), thickness=side_width,
                                     filter_nonmonotonic=False, filter_negative=False)

            support_lw = [wheel_point_X, wheel_point_y]
            support_rw = [corner_x, corner_y]
            line = np.array([support_lw, support_rw])
            line = self.add_3rd_dim(line)
            # print("border line", line)
            image = self.render_line(image, line, color=(0, 0, 255), thickness=side_width,
                                     filter_nonmonotonic=False, filter_negative=False)

            support_lw = [corner_x, corner_y]
            support_rw = [end_x, end_y]
            line = np.array([support_lw, support_rw])
            line = self.add_3rd_dim(line)
            # print("front line", line)
            image = self.render_line(image, line, color=(0, 0, 255), thickness=side_width,
                                     filter_nonmonotonic=False, filter_negative=False)

            support_lw = [corner_x, 0, corner_y]
            support_rw = [corner_x, corner_vertical_z, corner_y]
            line = np.array([support_lw, support_rw])
            line = self.add_3rd_dim(line, height_provided=True)
            # print("vertical line", line)
            image = self.render_line(image, line, color=(0, 0, 255), thickness=side_width,
                                     filter_nonmonotonic=False, filter_negative=False)

        # w, h, _ = image.shape
        # w2 = w // 2
        # h2 = h // 2
        # image[w2 - 2:w2 + 2, :] = (255, 0, 0)
        # image[:, h2 - 2:h2 + 2] = (255, 0, 0)

        return image


def main_revised():

    cfg_i = load_config()

    cfg = Namespace()
    cfg.__dict__ = cfg_i

    tv = TrajectoryVisualizer(cfg)

    cap = cv2.VideoCapture(0)
    cv2.destroyAllWindows()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def trackbar_to_rotation(trackbar_value):
        return radians(float(trackbar_value - 90 * TRACKBAR_GRANULARITY_FACTOR) / TRACKBAR_GRANULARITY_FACTOR)

    def get_frame_from_image():
        rgb = cv2.imread(cfg_i['file_path'])
        return rgb

    def loop():
        r = get_radius(0)
        background_img = get_frame_from_image()
        image = tv.render(background_img, r)
        image = cv2.resize(image, (1280, 720))
        cv2.imshow("image", image)

    def update_tvec_0(val):
        print("Adjusting tvec_0: %f" % val)
        tv.update_tvec(val / 100 - 0.1, 0)
        loop()

    def update_distance(val):
        print("Adjusting distance: %f" % val)
        tv.update_distance(val / 100 - 0.35)
        loop()

    def update_rvec_0(val):
        radians_val = trackbar_to_rotation(val)
        print("Adjusting rvec_0: %f" % radians_val)
        tv.update_rvec(radians_val, 0)
        loop()

    def update_rvec_1(val):
        radians_val = trackbar_to_rotation(val)
        print("Adjusting rvec_1: %f" % radians_val)
        tv.update_rvec(radians_val, 1)
        loop()

    def update_rvec_2(val):
        radians_val = trackbar_to_rotation(val)
        print("Adjusting rvec_2: %f" % radians_val)
        tv.update_rvec(radians_val, 2)
        loop()

    def export_rvec_values():
        rvec = tv.rvec
        tvec = tv.tvec
        tvec_cpy = tv.tvec_cpy
        log_file_path = cfg_i['file_path'].rsplit('/', 1)[0] + '/'
        log_file_name = cfg_i['file_path'].rsplit('/', 1)[1].rsplit('.', 1)[0] + '.RVEC_VALUES.log'

        f = open(log_file_path + log_file_name, "a")
        f.write(
            "Values determined at %s for RVEC: [%f %f %f]; TVEC varies so: origina: [%f %f %f] vs cpy: [%f %f %f]\n"
            % (datetime.date.today().strftime("%B %d, %Y"), rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2],
               tvec_cpy[0], tvec_cpy[1], tvec_cpy[2])
        )
        f.close()

        print("Values exported to %s" % (log_file_path + log_file_name))

    cv2.namedWindow('image')
    cv2.createTrackbar('rvec_0', 'image', 90 * TRACKBAR_GRANULARITY_FACTOR, 180 * TRACKBAR_GRANULARITY_FACTOR, update_rvec_0)
    cv2.createTrackbar('rvec_1', 'image', 90 * TRACKBAR_GRANULARITY_FACTOR, 180 * TRACKBAR_GRANULARITY_FACTOR, update_rvec_1)
    cv2.createTrackbar('rvec_2', 'image', 90 * TRACKBAR_GRANULARITY_FACTOR, 180 * TRACKBAR_GRANULARITY_FACTOR, update_rvec_2)
    cv2.createTrackbar('tvec_0 (cm - 10)', 'image', 10, 20, update_tvec_0)
    cv2.createTrackbar('tvec_0 (cm - 30)', 'image', 35, 70, update_distance)

    while True:
        loop()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 97:
            export_rvec_values()


def load_config():
    initial_cfg = {
        'center_color': (0, 255, 0),
        'center_width': 4,
        'side_color': (0, 255, 255),
        'side_width': 2,
        'curve_length': 30.0,
        'initial_steer': -1994.999999999999971,  # for this steer we have a straight line of trajectory
        'rvec': [0.04, 0.0, 0.0],
        # 'tvec': [0.0, 0.0, 0.0],
        'mark_count': 10,
        'start_dist': 6.0,
        'gap_dist': 1.0
    }

    with open("config.yaml", 'r') as stream:
        try:
            conf = yaml.load(stream)
            initial_cfg['file_path'] = conf['file']
            initial_cfg['tvec'] = np.array(conf['camera_position'])
            #initial_cfg['max_rvec_value'] = conf['max_rvec_value']
            initial_cfg['distance_mark'] = conf['distance_mark']
            initial_cfg['camera_matrix'] = np.array(conf['camera_matrix'])
            initial_cfg['distortion'] = conf['distortion']
        except yaml.YAMLError as exc:
            print(exc)

    return initial_cfg


if __name__ == "__main__":
    main_revised()
