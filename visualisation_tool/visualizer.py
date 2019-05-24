import pygame
import pygame_adaptor
from feed import Feed
from time import sleep
from map_handler import ImageWgsHandler

FRAME_DELAY = 0.005


class MapAdaptor:
    def __init__(self, map_path):
        self.wgs_handler = ImageWgsHandler(map_path)

    def convert_from_map_to_image(self, position, course):
        easting, northing = position
        x, y = self.wgs_handler.get_image_coord([easting], [northing])
        return x, y, course


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car visualisation")
        self.dimension = 800
        self.scale = 2
        self.exit = False

    def check_event_queue(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True

    def run(self):
        global FRAME_DELAY

        feed = Feed(
            "Resources/feeds/0ba94a1ed2e0449c.json",
            "Resources/feeds/0ba94a1ed2e0449c-0.mov",
            "Resources/feeds/0ba94a1ed2e0449c-seg"
        )
        map_handler = MapAdaptor(
            "Resources/data/high_res_full_UPB_hybrid.jpg"
        )

        segmentation, car_log = feed.fetch()
        _, position, course = car_log
        img_x, img_y, _ = map_handler.convert_from_map_to_image(position, course)
        prev_pos = (img_x, img_y)

        screen = pygame_adaptor.init_screen(self.dimension)
        map_image = pygame_adaptor.init_map(self.dimension, "Resources/data/high_res_full_UPB_hybrid.jpg", self.scale)
        pygame_adaptor.init_car(self.dimension, "Resources/images/car.png")

        while not self.exit:
            sleep(FRAME_DELAY)
            self.check_event_queue()

            segmentation, (_, position, course) = feed.fetch()
            img_x, img_y, angle = map_handler.convert_from_map_to_image(position, course)
            #bp = pygame_adaptor.crop_image(screen, map_image, (1672, 6000), self.dimension, scale_factor=self.scale)
            pygame_adaptor.blit_transform(
                screen,
                map_image,
                (0, 0),
                (0, 0),
                self.dimension,
                self.scale,
                angle
            )

            pygame_adaptor.update()
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
