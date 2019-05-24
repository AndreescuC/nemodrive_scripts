import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import tan, atan, radians
from skimage.draw import polygon_perimeter


def draw_projection(projection, path='Resources\drawings'):
    last_plot = max([int(entity.split("_")[-1].split(".")[0]) for entity in os.listdir(path) if os.path.isfile(entity)])

    projection = np.array([[(shade, shade, shade) for shade in row] for row in projection])

    plt.imshow(projection)
    plt.savefig(path + "plot_" + str(last_plot + 1) + ".png")


class SegmentationHandler:
    def __init__(self, camera_extrinsescs, camera_fov, max_seg_dist, camera_focal_point):
        self.camera_vertical_fov = camera_fov['vertical']
        self.camera_horizontal_fov = camera_fov['horizontal']
        self.max_seg_dist = max_seg_dist

        self.frame_depth = 800
        self.frame_width = None
        self.frame = self.compute_projection_frame()

        self.ce_translation = camera_extrinsescs[0]
        self.ce_rotation = camera_extrinsescs[1]

    def compute_projection_frame(self):
        self.frame_width = int(2 * self.frame_depth / tan(radians(90 - self.camera_horizontal_fov / 2)))

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

    def predict_projected(self, img_row, col_row, img_height, img_depth):

        # Predict real world y (depth)
        if img_row > img_height / 2:
            img_row = img_row - img_height / 2
        cam_projection_ratio = abs(2 * img_row / img_height)
        world_y = self.ce_translation['height'] /\
                  tan(atan(cam_projection_ratio * tan(self.camera_vertical_fov)) + self.ce_rotation['pitch'])

        world_x = 0

        return world_x, world_y

    def project_segmentation(self, segmentation):
        projection = deepcopy(self.frame)

        for row, row_elements in enumerate(segmentation):
            for column, pixel in enumerate(row_elements):

                # not part of segmentation
                if pixel == (0, 0, 0):
                    continue

                x, y = self.predict_projected(row, column)

                # grey segmentation
                projection[x][y] = 127

        draw_projection(projection)
        return projection


if __name__ == '__main__':
    h = SegmentationHandler()


