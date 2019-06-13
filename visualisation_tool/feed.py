import cv2
import glob
import json
from feed_vs import VisualiserFeed
from feed_pf import ParticleFilterFeed
from segmentation_projection import SegmentationHandler, read_camera_config


def capture_frames(video):
    frames = []

    vidcap = cv2.VideoCapture(video)
    ret, image = vidcap.read()
    while ret:
        frames.append(image)
        ret, image = vidcap.read()

    return frames


def read_segmentations(path):
    return [cv2.imread(file) for file in glob.glob(path + "/*.png")]


def project_segmentations(segmentation_frames):
    intrinsics, extrinsics, fov, resolution = read_camera_config()
    h = SegmentationHandler(intrinsics, extrinsics, fov, resolution, 60)

    return [h.project_segmentation(segmentation) for segmentation in segmentation_frames]


def extract_from_locations(data):
    return [
        {
            'tp': entry['timestamp'],
            'lat': entry['latitude'],
            'long': entry['longitude'],
            'course': entry['course']
        }
        for entry in data['locations']
    ]


def extract_from_phone(data):
    return [
        {
            'tp': x[0],
            'raw_lat': x[1],
            'raw_long': x[2]
        }
        for x in list(zip(data['phone_data']['tp'], data['phone_data']['latitude'], data['phone_data']['longitude']))
    ]


def extract_from_steer(data):
    return []


def extract_from_speed(data):
    return []


class Feed:
    def __init__(self, logs, videos, segmentation_paths):
        assert len(logs) == len(videos) == len(segmentation_paths)

        self.locations = []
        self.phone_data = []
        self.steer_data = []
        self.speed_data = []
        self.segmentation_frames = []
        self.video_frames = []

        for index in range(len(logs)):
            with open(logs[index]) as json_file:
                data = json.load(json_file)

                self.locations += extract_from_locations(data)
                self.phone_data += extract_from_phone(data)
                self.steer_data += extract_from_steer(data)
                self.speed_data += extract_from_speed(data)

            self.video_frames = capture_frames(videos[index])

            self.segmentation_frames = read_segmentations(segmentation_paths[index])
            self.segmentation_frames = project_segmentations(self.segmentation_frames)

        self.visualiser_feed = VisualiserFeed(self.locations, self.segmentation_frames)
        self.particle_filter_feed = ParticleFilterFeed(self.phone_data, self.segmentation_frames)

    def fetch_vs(self, tp=None):
        if tp is None:
            return self.visualiser_feed.fetch()

        is_done, feed_frame = self.visualiser_feed.fetch()
        while feed_frame['tp'] < tp and not is_done:
            is_done, feed_frame = self.visualiser_feed.fetch()

        return is_done, feed_frame

    def fetch_pf(self, tp=None):
        if tp is None:
            return self.particle_filter_feed.fetch()

        is_done, feed_frame = self.particle_filter_feed.fetch()
        while feed_frame['tp'] < tp and not is_done:
            is_done, feed_frame = self.particle_filter_feed.fetch()

        return is_done, feed_frame

    def get_freq_vs(self):
        return self.visualiser_feed.get_freq()

    def get_freq_pf(self):
        return self.particle_filter_feed.get_freq()

