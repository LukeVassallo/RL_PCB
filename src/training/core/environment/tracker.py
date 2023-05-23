import os

from core import video_utils
from collections import deque
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
matplotlib.use("Agg")

class tracker():
    def __init__(self, maxlen=1024):
        self.maxlen = maxlen

        self.all_comp_grids = deque(maxlen=self.maxlen)
        self.ratsnest = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)
        self.metrics = deque(maxlen=self.maxlen)
        self.frame_buffer = np.array([])

    def add_comp_grids(self, comp_grids=None):
        if comp_grids is not None:
            self.all_comp_grids.append(comp_grids)

    def get_last_comp_grids(self):
        return self.all_comp_grids[-1]

    def add(self, comp_grids=None, ratsnest=None):
        if comp_grids is not None:
            self.all_comp_grids.append(comp_grids)

        if ratsnest is not None:
            self.ratsnest.append(ratsnest)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_metrics(self, metrics):
        self.metrics.append(metrics)

    def reset(self):
        self.all_comp_grids.clear()
        self.ratsnest.clear()
        self.rewards.clear()
        self.metrics.clear()

    def create_video(self,
                     fileName=None,
                     v_id=None,
                     display_metrics=True,
                     fps=30):
        if display_metrics is True:
            video_utils.create_video(self.all_comp_grids,
                                     ratsnest=self.ratsnest,
                                     v_id=v_id,
                                     fileName=fileName,
                                     all_metrics=self.metrics,
                                     draw_debug=True,
                                     fps=fps)
        else:
            video_utils.create_video(self.all_comp_grids,
                                     ratsnest=self.ratsnest,
                                     v_id=v_id,
                                     fileName=fileName,
                                     all_metrics=None,
                                     draw_debug=True,
                                     fps=fps)

    def log_run_to_file(self, path=None, filename=None, kicad_pcb=None):
        f = open(os.path.join(path,filename), "w", encoding="utf-8")
        f.write(f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')}\r\n")
        f.write(f"module={self.__class__.__name__}\r\n")
        if kicad_pcb is not None:
            f.write(f".kicad_pcb={kicad_pcb}\r\n")

        # iterate over the number of components
        for i in range(len(self.metrics[0])):
            f.write("component begin\r\n")
            f.write(f"\tid={self.metrics[0][i]['id']}\r\n")
            f.write(f"\tname={self.metrics[0][i]['name']}\r\n")
            f.write(f"\tinitial_wirelength={self.metrics[0][i]['Wi']}\r\n")
            f.write(f"\tinitial_hpwl={self.metrics[0][i]['HPWLi']}\r\n")
            f.write(f"\texpert_wirelength={self.metrics[0][i]['We']}\r\n")
            f.write(f"\texpert_hpwl={self.metrics[0][i]['HPWLe']}\r\n")

            f.write("\theader begin\r\n")
            f.write("\t\tstep,wirelength,raw_wirelength,hpwl,raw_hpwl,overlap,weighted_cost,reward\r\n")
            f.write("\theader end\r\n")

            f.write("\tdata begin\r\n")
            # -1 due to done exiting after the first encounter.
            for j in range(len(self.metrics)-1):
                f.write(f"\t\t{j},{np.round(self.metrics[j][i]['W'],6)},{np.round(self.metrics[j][i]['raw_W'],6)},{np.round(self.metrics[j][i]['HPWL'],6)},{np.round(self.metrics[j][i]['raw_HPWL'],6)},{np.round(self.metrics[j][i]['ol'],6)},{np.round(self.metrics[j][i]['weighted_cost'],6)},{np.round(self.metrics[j][i]['reward'],6)},\r\n")
            f.write("data end\r\n")
        f.close()

    def create_plot(self, fileName=None):
        _, ax = plt.subplots(nrows=2, ncols=3)
        for i in range(len(self.metrics[0])):
            W = []
            raw_W = []
            HPWL = []
            raw_HPWL = []
            weighted_cost = []
            # -1 due to done exiting after the first encounter.
            for j in range(len(self.metrics)-1):
                W.append(self.metrics[j][i]["W"])
                raw_W.append(self.metrics[j][i]["raw_W"])

                HPWL.append(self.metrics[j][i]["HPWL"])
                raw_HPWL.append(self.metrics[j][i]["raw_HPWL"])

                weighted_cost.append(self.metrics[j][i]["weighted_cost"])

            ax[0,0].plot(W, label=f'{self.metrics[0][i]["name"]}')
            ax[1,0].plot(raw_W, label=f'{self.metrics[0][i]["name"]}, {np.round(self.metrics[0][i]["Wi"],2)}')
            ax[0,1].plot(HPWL, label=f'{self.metrics[0][i]["name"]}')
            ax[1,1].plot(raw_HPWL, label=f'{self.metrics[0][i]["name"]}, {np.round(self.metrics[0][i]["HPWLi"],2)}')
            ax[1,2].plot(weighted_cost, label=f'{self.metrics[0][i]["name"]}')

        ax[0,0].set_title("W")
        ax[1,0].set_title("raw_W")
        ax[0,1].set_title("HPWL")
        ax[1,1].set_title("raw_HPWL")
        ax[1,2].set_title("Weighted cost")
        ax[0,0].legend()
        ax[1,0].legend()
        ax[0,1].legend()
        ax[1,1].legend()
        ax[1,2].legend()
        plt.tight_layout()
        plt.savefig(fileName)
        plt.close()

    def capture_snapshot(self, fileName):
        video_utils.create_image(self.all_comp_grids,
                                 ratsnest=self.ratsnest,
                                 fileName=fileName,
                                 draw_debug=True)

    def video_tensor(self):
        return video_utils.get_video_tensor(self.all_comp_grids, self.ratsnest)
