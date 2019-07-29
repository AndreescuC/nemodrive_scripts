import cv2
import numpy as np
from math import cos, sin


NO_PARTICLES = 50
LOCATION_GAUSS_SPREAD = 30
ORIENTATION_GAUSS_SPREAD = 25
COLOR_MAPPINGS = {
    'black': [0, 0, 0],
    'segmentation': [0, 0, 255],
    'map': [0, 0, 255]
}


def __debug_mask_to_image__(array):
    img = np.zeros_like(array)

    for i, row in enumerate(array):
        for j, element in enumerate(row):
            if element == 0:     # empty
                img[i][j] = np.array([255, 255, 255])
            elif element == 1:   # road
                img[i][j] = np.array([180, 0, 0])
            elif element == 2:   # seg
                img[i][j] = np.array([0, 180, 0])
            elif element == 3:   # overlap
                img[i][j] = np.array([0, 0, 180])
            else:
                img[i][j] = np.array([0, 0, 0])

    cv2.imwrite("test.png", img)


def pad_segmentation(image):
    global COLOR_MAPPINGS
    h, w, c = image.shape
    img = np.vstack([image, np.full((h, w, c), COLOR_MAPPINGS['black'])])
    img = np.hstack((img,  np.full((h * 2, w // 2, c), COLOR_MAPPINGS['black'])))
    img = np.hstack((np.full((h * 2, w // 2, c), COLOR_MAPPINGS['black']), img))

    return img


def image_to_mask(img, color):
    return np.all(np.equal(img, color), axis=-1)


def mark_blit_point(image, point):
    x, y = point
    image[x][y] = (0, 255, 0)
    image[x][y-1] = (0, 255, 0)
    image[x][y+1] = (0, 255, 0)

    image[x-1][y] = (0, 255, 0)
    image[x-1][y-1] = (0, 255, 0)
    image[x-1][y+1] = (0, 255, 0)

    image[x+1][y] = (0, 255, 0)
    image[x+1][y-1] = (0, 255, 0)
    image[x+1][y+1] = (0, 255, 0)


def rotate(img, orientation):
    rows, cols, _ = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), orientation, 1)
    dst = cv2.warpAffine(img.astype(np.float64), M, (cols, rows))

    original_blit_point = (rows - 10, cols // 2)
    # mark_blit_point(img, original_blit_point)

    # y = y*cos(a) - x*sin(a)
    # x = y*sin(a) + x*cos(a)
    original_blit_point = (original_blit_point[0] - rows // 2, original_blit_point[1] - cols // 2)

    orientation = 360 - orientation
    blit_point = (
                int(original_blit_point[0] * cos(orientation) - original_blit_point[1] * sin(orientation)),
                int(original_blit_point[0] * sin(orientation) + original_blit_point[1] * cos(orientation))
        )
    blit_point = (blit_point[0] + rows // 2, blit_point[1] + cols // 2)
    # mark_blit_point(dst, blit_point)

    return dst, blit_point


def evaluate_segmentation_overlap(map_mask, segmentation, position, orientation):
    global COLOR_MAPPINGS

    rotated_seg, blit_point = rotate(segmentation, orientation)
    rotated_seg_mask = image_to_mask(rotated_seg, COLOR_MAPPINGS['segmentation'])

    h, w = rotated_seg_mask.shape

    map_mask[blit_point[0]:blit_point[0] + rotated_seg_mask.shape[0], blit_point[1]:blit_point[1] + rotated_seg_mask.shape[1]]\
        += rotated_seg_mask

    return sum(map_mask) / (w * h)


class Assessor:
    def __init__(self):
        self.methods = ['segmentation_overlap']

    def asses(self, particles, map_mask, segmentation_mask, method='segmentation_overlap'):
        assert method in self.methods

        # Transform from 3-channels images to 1-channel encoded masks
        global COLOR_MAPPINGS
        map_mask = image_to_mask(map_mask, COLOR_MAPPINGS['map'])

        # segmentation_mask = pad_segmentation(segmentation_mask)

        # Compute new weights
        weight_sum = 0
        particle: Particle
        for particle in particles:
            particle.weight = evaluate_segmentation_overlap(
                map_mask,
                segmentation_mask,
                particle.position,
                particle.orientation
            )
            weight_sum += particle.weight

        # Normalize distribution
        for particle in particles:
            particle.weight /= weight_sum


class Particle:
    def __init__(self, position, orientation, no_particle):
        self.position = position
        self.orientation = orientation

        self.weight = 1 / no_particle


def init_particles(location, course, no_particles):
    global LOCATION_GAUSS_SPREAD, ORIENTATION_GAUSS_SPREAD

    return [
        Particle(
            (np.random.normal(location[0], LOCATION_GAUSS_SPREAD), np.random.normal(location[1], LOCATION_GAUSS_SPREAD)),
            np.random.normal(course, ORIENTATION_GAUSS_SPREAD),
            no_particles
        )
        for _ in range(no_particles)
    ]


if __name__ == '__main__':
    map_mask = cv2.imread("/home/andi/Downloads/pixil-frame-0 (1).png")
    segmentation_mask = cv2.imread("/home/andi/Downloads/pixil-frame-0.png")

    particle = Particle((40, 40), 90, 1)

    assessor = Assessor()
    assessor.asses([particle], map_mask, segmentation_mask)
