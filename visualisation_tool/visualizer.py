import cv2
import pygame
import pygame_adaptor
import random
import io_handler
from feed import Feed
from time import sleep
from math import cos, tan, sqrt
from map_handler import ImageWgsHandler
from particle_filter import init_particles, NO_PARTICLES, Assessor


class MapAdaptor:
    def __init__(self, map_path):
        self.wgs_handler = ImageWgsHandler(map_path)

    def convert_from_map_to_image(self, position, course):
        easting, northing = position
        x, y = self.wgs_handler.get_image_coord([easting], [northing], convert_method=2)
        return int(x[0]), int(y[0]), course - 90

    def convert_to_snippet(self, x, y, pos, image_rotation):
        image_center_x, image_center_y = pos
        l2 = image_center_x - x
        l1 = y - image_center_y

        a1 = (l1 - l2 * tan(image_rotation)) * cos(image_rotation)
        a2 = sqrt(l2 ** 2 + l1 ** 2 - a1 ** 2)

        return -a2, a1


class Game:
    def __init__(self, params):
        # Cannot display a heat map without particle filter algorithm
        assert not (params['heat_map'] and not params['particle_filter'])

        self.is_pf = params['particle_filter']
        self.is_hm = params['heat_map']
        self.frame_delay = params['frame_delay']

        pygame.init()
        pygame.display.set_caption("Car visualisation")

        self.dimension = 800
        self.scale = 20
        self.exit = False

    def check_event_queue(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True

    def run(self):
        feed = Feed(
            ["Resources/feeds/0ba94a1ed2e0449c.json"],
            ["Resources/feeds/0ba94a1ed2e0449c-0.mov"],
            ["Resources/feeds/0ba94a1ed2e0449c-seg"]
        )
        map_handler = MapAdaptor(
            "Resources/data/high_res_full_UPB_hybrid.jpg"
        )
        assessor = Assessor()

        _, segmentation, log = feed.fetch_vs()
        geographical_position = (log['lat'], log['long'])
        course = log['course']

        img_y, img_x, angle = map_handler.convert_from_map_to_image(geographical_position, course)
        pos = (img_x, img_y)

        _, filter_log = feed.fetch_pf(log['tp'])

        # TODO: replace these with commented raw location
        raw_lat = log['lat'] - random.uniform(0.00005, 0.00015)    # filter_log['lat']
        raw_long = log['long'] - random.uniform(0.00005, 0.00015)  # filter_log['long']

        screen = pygame_adaptor.init_screen(self.dimension)
        map_image, real_size = pygame_adaptor.init_map(self.dimension, "Resources/data/high_res_full_UPB_hybrid.jpg", self.scale)
        car_screen_position = pygame_adaptor.init_car(self.dimension, "Resources/images/car.png")

        seg_sprite = pygame_adaptor.display_segmentation(car_screen_position, segmentation)

        while not self.exit:
            self.check_event_queue()
            sleep(self.frame_delay)

            current_map_image = pygame_adaptor.blit_transform(
                screen,
                map_image,
                pos,
                real_size,
                self.dimension,
                self.scale,
                angle
            )

            raw_img_y, raw_img_x, angle = map_handler.convert_from_map_to_image((raw_lat, raw_long), course)
            x_from_center, y_from_center = map_handler.convert_to_snippet(raw_img_x, raw_img_y, pos, angle)
            particles = init_particles(
                (x_from_center + self.dimension / 2, y_from_center + self.dimension / 2),
                angle,
                NO_PARTICLES
            )
            assessor.asses(particles, current_map_image, cv2.imread(segmentation))

            _, segmentation, log = feed.fetch_vs()
            geographical_position = (log['lat'], log['long'])
            course = log['course']

            img_y, img_x, angle = map_handler.convert_from_map_to_image(geographical_position, course)
            pos = (img_x, img_y)
            seg_sprite = pygame_adaptor.display_segmentation(car_screen_position, segmentation, seg_sprite)

            pygame_adaptor.update()
        pygame.quit()


if __name__ == '__main__':
    params = io_handler.get_parameters()
    game = Game(params)
    game.run()
