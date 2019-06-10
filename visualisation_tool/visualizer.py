import pygame
import pygame_adaptor
from feed import Feed
from time import sleep
from map_handler import ImageWgsHandler

FRAME_DELAY = 0.2
USE_OLD_PNGS = True


class MapAdaptor:
    def __init__(self, map_path):
        self.wgs_handler = ImageWgsHandler(map_path)

    def convert_from_map_to_image(self, position, course):
        easting, northing = position
        x, y = self.wgs_handler.get_image_coord([easting], [northing], convert_method=2)
        return int(x[0]), int(y[0]), course - 90


class Game:
    def __init__(self):
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
        global FRAME_DELAY, USE_OLD_PNGS

        feed = Feed(
            "Resources/feeds/0ba94a1ed2e0449c.json",
            "Resources/feeds/0ba94a1ed2e0449c-0.mov",
            "Resources/feeds/0ba94a1ed2e0449c-seg",
            "Resources/feeds/png",
            USE_OLD_PNGS
        )
        map_handler = MapAdaptor(
            "Resources/data/high_res_full_UPB_hybrid.jpg"
        )

        _, segmentation, (_, geographical_position, course) = feed.fetch()
        img_y, img_x, angle = map_handler.convert_from_map_to_image(geographical_position, course)
        pos = (img_x, img_y)

        screen = pygame_adaptor.init_screen(self.dimension)
        map_image, real_size = pygame_adaptor.init_map(self.dimension, "Resources/data/high_res_full_UPB_hybrid.jpg", self.scale)
        car_screen_position = pygame_adaptor.init_car(self.dimension, "Resources/images/car.png")

        seg_sprite = pygame_adaptor.display_segmentation(car_screen_position, segmentation)

        while not self.exit:
            self.check_event_queue()
            sleep(FRAME_DELAY)

            pygame_adaptor.blit_transform(
                screen,
                map_image,
                pos,
                real_size,
                self.dimension,
                self.scale,
                angle
            )

            _, segmentation, (_, geographical_position, course) = feed.fetch()
            img_y, img_x, angle = map_handler.convert_from_map_to_image(geographical_position, course)
            pos = (img_x, img_y)
            seg_sprite = pygame_adaptor.display_segmentation(car_screen_position, segmentation, seg_sprite)

            pygame_adaptor.update()
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
