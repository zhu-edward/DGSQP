
import numpy as np

import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from DGSQP.tracks.track_lib import get_track
from globals import TRACK

track_obj = get_track(TRACK).get_track_segment([60, 80], 10)

base_dir = pathlib.Path(pathlib.Path.home(), 'results')

# M = 50 all
# data_dir = 'comparison_study_kinematic_2024-04-16_11-01-13'

# data_dir = 'ablation_study_kinematic_2024-04-17_09-17-41'

data_dir = 'comparison_study_f1_dynamic_2024-04-21_14-35-43'

_data_dir = pathlib.Path(base_dir, data_dir)
scenarios = os.listdir(_data_dir)
scenarios.sort()

for s in scenarios:
    s_path = pathlib.Path(_data_dir, s)
    if os.path.isdir(s_path):
        out_path = pathlib.Path(s_path, 'out')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        files = os.listdir(s_path)
        samples = 0
        success = []
        x0, y0 = [], []
        status = []
        for f in files:
            if f.endswith('.pkl'):
                samples += 1
                f_path = pathlib.Path(s_path, f)
                with open(f_path, 'rb') as _f:
                    d = pickle.load(_f)
                msg = d['solver_result']['msg']
                if 'path' in s and msg == 'MCP_Solved':
                    success.append(True)
                elif 'dgsqp' in s and msg == 'conv_abs_tol':
                    success.append(True)
                else:
                    success.append(False)
                status.append(msg)

                initial_condition = d['initial_condition']
                x0.append(np.mean([s.x.x for s in initial_condition]))
                y0.append(np.mean([s.x.y for s in initial_condition]))

        x0 = np.array(x0)
        y0 = np.array(y0)

        if samples > 0:
            print(f'{s}: {np.sum(success)}/{samples}')

        fig_success = plt.figure(figsize=(10,10))
        ax_success = fig_success.gca()
        track_obj.plot_map(ax_success)
        succ = np.where(success)[0].astype(int)
        fail = np.setdiff1d(np.arange(len(success)), succ).astype(int)
        ax_success.scatter(x0[succ], y0[succ], facecolors='none', edgecolors='g', marker='o', s=40)
        ax_success.scatter(x0[fail], y0[fail], c='r', marker='x', s=40)
        ax_success.set_aspect('equal')
        ax_success.get_xaxis().set_ticks([])
        ax_success.get_yaxis().set_ticks([])
        ax_success.set_title(s)
        fig_success.savefig(pathlib.Path(out_path, f'{s}_success.png'), dpi=200, bbox_inches='tight')

        fig_status = plt.figure(figsize=(10, 10))
        ax_status = fig_status.gca()
        ax_status.hist(status)
        ax_status.set_title(s)
        fig_status.savefig(pathlib.Path(out_path, f'{s}_status.png'), dpi=200, bbox_inches='tight')
