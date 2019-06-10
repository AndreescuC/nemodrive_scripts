import os
import cv2
import glob
import json
import numpy as np
from segmentation_projection import SegmentationHandler, read_camera_config


FPS = 3
TP_ERROR = 50  # ms


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


def get_parsed_logs(log):
    with open(log) as json_file:
        data = json.load(json_file)
        steering = [
            entry['course']
            for entry in data['locations']
        ]
        locations = [
            (entry['latitude'], entry['longitude'], entry['timestamp'])
            for entry in data['locations']
        ]

    assert len(locations) == len(steering)
    # Keeping only 3 FPS (from 30 FPS)
    return [
        (location[2], (location[0], location[1]), steer)
        for location, steer in zip(locations[::10], steering[::10])
    ]


def create_segmentations_dir(seg_dir):
    if os.path.exists(seg_dir):
        for file in os.listdir(seg_dir):
            file_path = os.path.join(seg_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    else:
        os.mkdir(seg_dir)


class Feed:
    def __init__(self, log, video, segmentation_path, segmentation_store_dir, use_old_pngs):
        self.seg_dir = segmentation_store_dir

        self.logs = get_parsed_logs(log)
        self.video_frames = capture_frames(video)
        self.segmentation_frames = read_segmentations(segmentation_path)
        self.segmentation_frames = self.transform_segmentations()
        self.segmentation_frames = self.store_as_pngs(use_old_pngs)

        self.sync_logs()

        assert len(self.video_frames) == len(self.segmentation_frames) == len(self.logs)
        self.index = -1

    def fetch(self):
        self.index += 1
        if self.index >= len(self.logs):
            self.index = 0
        return self.index == 0, self.segmentation_frames[self.index], self.logs[self.index]

    def blind_fetch(self):
        index = self.index
        feed_done, segmentation, log = self.fetch()
        self.index = index
        return feed_done, segmentation, log

    def sync_logs(self):
        global FPS, TP_ERROR

        starting_timestamp = self.logs[0][0]
        increment = 1000 / FPS
        timestamps = [starting_timestamp + frame * increment for frame in range(len(self.video_frames))]

        logs_idx = frames_idx = 0
        logs_selection = []
        while logs_idx < len(self.logs) and frames_idx < len(self.video_frames):
            if abs(timestamps[frames_idx] - self.logs[logs_idx][0]) < TP_ERROR:
                logs_selection.append(self.logs[logs_idx])
                frames_idx += 1

            logs_idx += 1

        # add last log
        if len(logs_selection) == len(self.video_frames) - 1:
            if logs_idx == len(self.logs):
                logs_selection.append(self.logs[logs_idx - 1])
            else:
                logs_selection.append(self.logs[logs_idx])

        self.logs = logs_selection

    def transform_segmentations(self):
        assert self.segmentation_frames

        intrinsics, extrinsics, fov, resolution = read_camera_config()
        h = SegmentationHandler(intrinsics, extrinsics, fov, resolution, 60)

        return [h.project_segmentation(segmentation) for segmentation in self.segmentation_frames]

    def store_as_pngs(self, use_old_pngs = False):
        if use_old_pngs:
            return [os.path.join(self.seg_dir, "segmentation_%d.png" % i) for i in range(len(self.segmentation_frames))]

        create_segmentations_dir(self.seg_dir)
        for i, segmentation in enumerate(self.segmentation_frames):
            b_channel, g_channel, r_channel = cv2.split(segmentation)

            alpha_channel = np.array([
                [255 if sum(element) else 0 for element in row]
                for row in segmentation
            ], dtype=b_channel.dtype)
            alpha_segmentation = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            cv2.imwrite(os.path.join(self.seg_dir, "segmentation_%d.png" % i), alpha_segmentation)

        return [os.path.join(self.seg_dir, "segmentation_%d.png" % i) for i in range(len(self.segmentation_frames))]
