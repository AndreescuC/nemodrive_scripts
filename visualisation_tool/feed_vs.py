import os
import cv2
import numpy as np
from fractions import Fraction


def create_segmentation_dir(seg_dir):
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


def store_as_pngs(segmentation_frames, seg_dir, use_old_pngs = False):
    if use_old_pngs:
        return [os.path.join(seg_dir, "segmentation_%d.png" % i) for i in range(len(segmentation_frames))]

    create_segmentation_dir(seg_dir)
    for i, segmentation in enumerate(segmentation_frames):
        b_channel, g_channel, r_channel = cv2.split(segmentation)

        alpha_channel = np.array([
            [255 if sum(element) else 0 for element in row]
            for row in segmentation
        ], dtype=b_channel.dtype)
        alpha_segmentation = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        cv2.imwrite(os.path.join(seg_dir, "segmentation_%d.png" % i), alpha_segmentation)

    return [os.path.join(seg_dir, "segmentation_%d.png" % i) for i in range(len(segmentation_frames))]


class VisualiserFeed:
    def __init__(self, location_logs, segmentation_frames):
        self.logs = location_logs
        self.segmentation_frames = store_as_pngs(
            segmentation_frames=segmentation_frames,
            seg_dir="Resources/feeds/png",
            use_old_pngs=True
        )

        self.fps = 3
        self.sync_permissible_tp_error = 50  # ms

        self.sync_logs()

        assert len(self.segmentation_frames) == len(self.logs)
        self.index = -1

    def fetch(self):
        self.index += 1
        if self.index >= len(self.logs):
            self.index = 0
        return self.index == 0, self.segmentation_frames[self.index], self.logs[self.index]

    def get_freq(self):
        return Fraction(self.fps)

    def sync_logs(self):
        starting_timestamp = self.logs[0]['tp']
        increment = 1000 / self.fps
        timestamps = [starting_timestamp + frame * increment for frame in range(len(self.segmentation_frames))]

        logs_idx = frames_idx = 0
        logs_selection = []
        while logs_idx < len(self.logs) and frames_idx < len(self.segmentation_frames):
            if abs(timestamps[frames_idx] - self.logs[logs_idx]['tp']) < self.sync_permissible_tp_error:
                logs_selection.append(self.logs[logs_idx])
                frames_idx += 1

            logs_idx += 1

        # add last log
        if len(logs_selection) == len(self.segmentation_frames) - 1:
            if logs_idx == len(self.logs):
                logs_selection.append(self.logs[logs_idx - 1])
            else:
                logs_selection.append(self.logs[logs_idx])

        self.logs = logs_selection
