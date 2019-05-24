import pygame
import pygame_functions as pygf


def update():
    pygf.updateDisplay()


def init_screen(dimension):
    screen = pygf.screenSize(dimension, dimension)
    pygf.setAutoUpdate(False)

    return screen


def init_map(dimension, img_path, scale_factor=2):
    # image = pygame.Surface((dimension, dimension))
    map_image = pygame.image.load(img_path).convert()
    map_image = pygame.transform.scale(map_image, (dimension * scale_factor, dimension * scale_factor))
    # image.blit(map_image, (1, 1))

    return map_image


def init_car(dimension, img_path):
    car_sprite = pygf.makeSprite(img_path)
    pygf.moveSprite(car_sprite, dimension / 2, dimension / 2, True)
    pygf.transformSprite(car_sprite, -90, 1)
    pygf.showSprite(car_sprite)


def crop_image(screen, image, real_img_pos, dimension, real_img_dim=(9728, 10496), scale_factor=1):
    blit_point = (
        dimension * (real_img_pos[0] * scale_factor / real_img_dim[0] - 0.5),
        dimension * (real_img_pos[1] * scale_factor / real_img_dim[1] - 0.5)
    )

    screen.blit(image, (0, 0), (*blit_point, dimension, dimension))

    return blit_point


def blit_transform(surf, image, pos, origin_pos, dimension, scale_factor, angle):
    w, h = image.get_size()
    scaled_origin_pos = (origin_pos[0] * scale_factor, origin_pos[0] * scale_factor)

    # compute the axis aligned bounding box of the rotated image
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # compute the translation of the pivot
    pivot = pygame.math.Vector2(scaled_origin_pos[0], -scaled_origin_pos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # compute blit-ing point
    width_from_center = dimension / 2 - pos[0]
    height_from_center = dimension / 2 - pos[0]
    origin = (
        pos[0] - scaled_origin_pos[0] + min_box[0] - pivot_move[0] + width_from_center,
        pos[1] - scaled_origin_pos[1] - max_box[1] + pivot_move[1] + height_from_center
    )

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)

    # rotate and blit the image
    surf.blit(rotated_image, origin)
