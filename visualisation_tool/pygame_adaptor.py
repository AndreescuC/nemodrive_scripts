import cv2
import pygame
import numpy as np
import pygame_functions as pygf


def update():
    pygf.updateDisplay()


def init_screen(dimension):
    screen = pygf.screenSize(dimension, dimension)
    pygf.setAutoUpdate(False)

    return screen


def init_map(dimension, img_path, scale_factor=2):
    # image = pygame.Surface((dimension, dimension))
    map_image = pygame.image.load(img_path).convert_alpha()
    original_size = map_image.get_size()
    map_image = pygame.transform.scale(map_image, (dimension * scale_factor, dimension * scale_factor))
    # image.blit(map_image, (1, 1))

    return map_image, original_size


def write_segmentation_png(segmentation):
    image = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGBA)
    image[np.all(image == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

    return image


def init_car(dimension, img_path):
    car_sprite = pygf.makeSprite(img_path)
    position = (dimension / 2, dimension / 2)
    pygf.moveSprite(car_sprite, position[0], position[1], True)
    pygf.transformSprite(car_sprite, -90, 0.5)
    pygf.showSprite(car_sprite)

    return position


def display_segmentation(position, img_path, old_sprite=None):
    seg_sprite = pygf.makeSprite(img_path)
    pygf.moveSprite(seg_sprite, position[0], position[1] - 170, True)
    pygf.transformSprite(seg_sprite, 0, 0.9)

    if old_sprite:
        pygf.hideSprite(old_sprite)

    pygf.showSprite(seg_sprite)
    return seg_sprite


def crop_image(screen, image, real_img_pos, dimension, real_img_dim=(9728, 10496), scale_factor=1):
    blit_point = (
        dimension * (real_img_pos[0] * scale_factor / real_img_dim[0] - 0.5),
        dimension * (real_img_pos[1] * scale_factor / real_img_dim[1] - 0.5)
    )

    screen.blit(image, (0, 0), (*blit_point, dimension, dimension))

    return blit_point


def scale_position(pos, scale, frame_dim, real_shape):
    return frame_dim * pos[0] * scale / real_shape[0], frame_dim * pos[1] * scale / real_shape[1]


def blit_transform(surf, image, origin_pos, real_shape, dimension, scale_factor, angle):
    w, h = image.get_size()

    crop_margin = 300

    # pos = scale_position(pos, scale_factor, dimension, real_shape)
    origin_pos = scale_position(origin_pos, scale_factor, dimension, real_shape)

    crop_origin = [origin_pos[i] - dimension / 2 - crop_margin if origin_pos[i] >= dimension / 2 + crop_margin else 0 for i in range(2)]
    max_crop = (min(dimension + 2 * crop_margin, w - crop_origin[0]), min(dimension + 2 * crop_margin, h - crop_origin[1]))
    cropped_image = image.subsurface(pygame.Rect(crop_origin, max_crop))


    cropped_w, cropped_h = max_crop
    cropped_origin_pos = (cropped_w / 2, cropped_h / 2)

    # compute the axis aligned bounding box of the rotated image
    box = [pygame.math.Vector2(p) for p in [(0, 0), (cropped_w, 0), (cropped_w, -cropped_h), (0, -cropped_h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # compute the translation of the pivot
    pivot = pygame.math.Vector2(cropped_w / 2, -cropped_h / 2)
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # compute blit-ing point
    origin = (
        min_box[0] - pivot_move[0] - crop_margin,
        -max_box[1] + pivot_move[1] - crop_margin
    )

    rotated_image = pygame.transform.rotate(cropped_image, angle)
    surf.blit(rotated_image, origin)

    return pygame.surfarray.array3d(rotated_image)

