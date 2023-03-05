#import sys
#sys.path.append('/home/luke/Desktop/semi_autonomous/py/pcb_component_w_vec_distance_v2')
import os, sys
sys.path.append(os.path.join(os.environ["RL_PCB"], "pcb_component_w_vec_distance_v2"))
from core import video_utils

from collections import deque
import numpy as np

# deque perform memory-efficient appends and pops from either size with approx O(1). On the other hand lists are optimized for fixed-length operations and incur O(n) movement cost for pop and insert.

# If deque is of fixed length, appends that exceed its length will cause the left-most element to be pushed out.

# eg.
# d = deque(maxlen=5)
# len(d)    # 0
# for i in range(10): d.append(i)
# for i in range(5): print(d[i])        # 5,6,7,8,9

class tracker():
    def __init__(self, maxlen=1024):
        self.maxlen = maxlen

        self.all_comp_grids = deque(maxlen=self.maxlen)
        self.ratsnest = deque(maxlen=self.maxlen)

        self.frame_buffer = np.array([])

    def add_observation(self, comp_grids=None, los_grids=None, los=None, ol_grids=None, ol=None):
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
            self.frame_buffer = video_utils.video_frames(self.all_comp_grids, self.ratsnest, v_id=v_id)
        else:
            self.frame_buffer = np.concatenate((self.frame_buffer,video_utils.video_frames(self.all_comp_grids, self.ratsnest, v_id=v_id)), axis=0)

    def write_frame_buffer(self, fileName=None, reset=False):
        video_utils.write_frame_buffer(self.frame_buffer)
        self.frame_buffer = np.array([])    

       
    
  
        
