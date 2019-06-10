import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan, atan, radians, pi
from skimage.draw import polygon_perimeter


def read_camera_config():
    with open("config.yaml", 'r') as stream:

        try:
            conf = yaml.load(stream)
            return \
                conf['camera_intrinsics'],\
                conf['camera_extrinsics'],\
                conf['camera_specs']['fov'],\
                conf['camera_specs']['resolution']

        except yaml.YAMLError as exc:
            print(exc)
            return None, None, None, None


def get_last_plot(path):
    files = [int(entity.split("_")[-1].split(".")[0]) for entity in os.listdir(path) if os.path.isfile(entity)]
    return max(files) if files else 0


def draw_segmentation(segmentation, path='Resources/drawings'):
    last_plot = get_last_plot(path)

    plt.imshow(segmentation)
    plt.show()
    plt.savefig(path + "seg_" + str(last_plot + 1) + ".png")


def draw_projection(projection, path='Resources/drawings'):
    last_plot = get_last_plot(path)

    projection = np.array([[(shade, shade, shade) for shade in row] for row in projection])

    plt.imshow(projection)
    plt.show()
    plt.savefig(path + "project_" + str(last_plot + 1) + ".png")


class SegmentationHandler:
    def __init__(self, camera_intrinsics, camera_extrinsics, camera_fov, resolution, max_seg_dist):
        self.camera_vertical_fov = radians(camera_fov['vertical'])
        self.camera_horizontal_fov = radians(camera_fov['horizontal'])
        self.max_seg_dist = max_seg_dist

        self.slope_factor = 0.1
        self.distance_factor = 0.4
        self.offset = 10

        self.frame_depth = 800
        self.frame_width = None
        self.frame = self.compute_projection_frame()

        self.pixels_to_meter = self.frame_depth / self.max_seg_dist

        self.real_camera_resolution = resolution

        self.ci_distortion = camera_intrinsics['distortion']
        self.ci_camera_matrix = camera_intrinsics['camera_matrix']
        self.ci_aux_inv_M = np.linalg.inv(self.ci_camera_matrix)

        self.ce_translation = np.array(list(camera_extrinsics['translation'].values())).reshape(3, 1)
        self.ce_rotation = np.array([radians(v) for v in camera_extrinsics['rotation'].values()]).reshape(3, 1)
        self.ce_rotation_matrix = cv2.Rodrigues(np.float32(
            list(camera_extrinsics['rotation'].values())
        ))[0]
        self.ce_aux_inv_R = np.linalg.inv(self.ce_rotation_matrix)

    def compute_projection_frame(self):
        self.frame_width = int(2 * self.frame_depth / tan(pi / 2 - self.camera_horizontal_fov / 2))

        img = np.full(shape=(self.frame_depth, self.frame_width), fill_value=255, dtype=np.uint8)
        rr, cc = polygon_perimeter(
            [0, 0, self.frame_depth],
            [0, self.frame_width, self.frame_width / 2],
            shape=img.shape,
            clip=True
        )
        # black fov border
        img[rr, cc] = 0

        return img

    def get_intrinsic_matrix(self, scaled_width, scaled_height):
        K = np.zeros((3, 4))
        K[:, :-1] = self.ci_camera_matrix

        S = np.array([
            [scaled_width / 1920, 0, 0],
            [0, scaled_height / 1080, 0],
            [0, 0, 1]
        ])
        return np.matmul(S, K)

    def get_homography_matrix(self, scaled_width, scaled_height):
        K = self.get_intrinsic_matrix(scaled_width, scaled_height)
        M = self.get_transformation_matrix(K=K)
        H = np.matmul(K, M)
        return H

    def get_transformation_matrix(self, K):
        M = np.array([
            [1.0, 0.0, 0.0, 0],
            [0.0, 1.0, 0.0, 0],
            [0.0, 0.0, 1.0, K[0, 0] * self.distance_factor],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # translation matrix
        T = np.array([
            [1, 0, -K[0, 2]],
            [0, 1, -K[1, 2] - self.offset],
            [0, 0, 0],
            [0, 0, 1]
        ])

        # rotation matrix
        alpha = - pi / (2 + self.slope_factor)
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1]
        ])

        # extrinsic matrix
        return np.matmul(M, np.matmul(Rx, T))

    def project_segmentation(self, segmentation):
        orig_h, orig_w, _ = segmentation.shape

        H = self.get_homography_matrix(orig_w, orig_h)

        # RGB to BGR and crop
        screen = segmentation[:, :, ::-1]
        screen = screen[:, :, ::-1]

        w, h, _ = screen.shape

        return cv2.warpPerspective(screen, H, (h, w), flags=cv2.WARP_INVERSE_MAP)

    def to_pixel_width(self, x, upper_limit):
        midpoint = upper_limit // 2
        x = int(self.pixels_to_meter * x)
        x += midpoint

        if x >= upper_limit:
            x = upper_limit - 1

        return x if x > 0 else 0

    def to_pixel_height(self, y, upper_limit):
        y = int(self.pixels_to_meter * y)
        y = upper_limit - y
        if y >= upper_limit:
            y = upper_limit - 1

        return y if y > 0 else 0

    def convert_real_resolution(self, img_col, img_row, segmentation_h, segmentation_w):
        real_width = self.real_camera_resolution['width'] * img_col / segmentation_w
        real_height = self.real_camera_resolution['height'] * img_row / segmentation_h

        return real_width, real_height


if __name__ == '__main__':
    intrinsics, extrinsics, fov, resolution = read_camera_config()
    h = SegmentationHandler(intrinsics, extrinsics, fov, resolution, 60)

    seg_image = cv2.imread("Resources/feeds/0ba94a1ed2e0449c-seg/99.png")
    draw_segmentation(seg_image)

    projection = h.project_segmentation(seg_image)
    cv2.imshow("test", projection)
    cv2.waitKey(0)

