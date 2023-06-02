from core import video_utils

from collections import deque
import numpy as np

class tracker():
    def __init__(self, maxlen=1024):
        self.maxlen = maxlen

        self.all_comp_grids = deque(maxlen=self.maxlen)
        self.ratsnest = deque(maxlen=self.maxlen)

        self.frame_buffer = np.array([])

    def add_observation(self, comp_grids=None,
                        los_grids=None,
                        los=None,
                        ol_grids=None,
                        ol=None):
        if comp_grids is not None:
            self.all_comp_grids.append(comp_grids)

    def add_ratsnest(self, ratsnest):
        if ratsnest is not None:
            self.ratsnest.append(ratsnest)

    def reset(self):
        self.all_comp_grids.clear()
        self.ratsnest.clear()

    def create_video(self, fileName=None, v_id=None):
        video_utils.create_video(self.all_comp_grids, self.ratsnest, v_id=v_id)

    def update_frame_buffer(self, v_id=None):
        if self.frame_buffer.size == 0:
            self.frame_buffer = video_utils.video_frames(self.all_comp_grids,
                                                         self.ratsnest,
                                                         v_id=v_id)
        else:
            self.frame_buffer = np.concatenate(
                (self.frame_buffer, video_utils.video_frames(self.all_comp_grids, self.ratsnest, v_id=v_id)),
                 axis=0)

    def write_frame_buffer(self, fileName=None, reset=False):
        video_utils.write_frame_buffer(self.frame_buffer)
        self.frame_buffer = np.array([])
