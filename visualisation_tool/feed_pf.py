import math
from fractions import Fraction


class ParticleFilterFeed:
    def __init__(self, raw_localization_data, segmentation_frames):
        self.logs = raw_localization_data
        self.raw_location_length = len(raw_localization_data)

        self.segmentation_frames = segmentation_frames
        self.segmentation_length = len(segmentation_frames)

        # How many localization entries correspond to one road segmentation entry
        self.frequency_ratio = math.ceil(self.raw_location_length / self.segmentation_length)

        self.index = -1

    def fetch(self):
        """"Fetches current feed frame

        :return (whether the feed reached its end, the current feed frame)
        :rtype (bool, dict)

        """

        self.index += 1
        if self.index >= self.raw_location_length:
            self.index = 0
        return self.index == 0, self.compose_entry()

    def get_freq(self):
        return Fraction(self.raw_location_length, 36)

    def compose_entry(self):
        segmentation_index = self.index // self.frequency_ratio
        current_log = self.logs[self.index]

        return {
            'tp': current_log['tp'],
            'lat': current_log['raw_lat'],
            'long': current_log['raw_long'],
            'seg': self.segmentation_frames[segmentation_index],
        }
