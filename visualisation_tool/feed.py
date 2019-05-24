import cv2
import glob
import json


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


class Feed:
    def __init__(self, log, video, segmentation_path):
        self.logs = get_parsed_logs(log)
        self.video_frames = capture_frames(video)
        self.segmentation_frames = read_segmentations(segmentation_path)

        self.sync_logs()

        assert len(self.video_frames) == len(self.segmentation_frames) == len(self.logs)
        self.index = -1

    def fetch(self):
        self.index += 1
        if self.index >= len(self.logs):
            self.index = 0
        return self.segmentation_frames[self.index], self.logs[self.index]

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

