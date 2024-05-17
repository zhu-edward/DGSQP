
import numpy as np

import os
import pathlib
import pickle
import pdb

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.cm import ScalarMappable

from DGSQP.tracks.track_lib import get_track
from globals import TRACK

track_obj = get_track(TRACK)

base_dir = pathlib.Path(pathlib.Path.home(), 'results')

# data_dir = 'regularization_study_barc_kinematic_2024-04-23_09-26-58'
data_dir = 'regularization_study_barc_kinematic_2024-04-26_08-20-50'

_data_dir = pathlib.Path(base_dir, data_dir)
scenarios = os.listdir(_data_dir)
scenarios.sort()

eval_type = 'always'
# eval_type = 'once'

_success_rate = dict()
_solve_time = dict()
reg = set()
decay = set()

out_path = pathlib.Path(_data_dir, 'out')
if not os.path.exists(out_path):
    os.makedirs(out_path)

for s in scenarios:
    s_path = pathlib.Path(_data_dir, s)
    if os.path.isdir(s_path):
        if eval_type not in s:
            continue
        if s.find('reg_') < 0:
            _reg = 0
        else:
            reg_i = s.find('reg_') + 4
            reg_j = s.find('-', reg_i)
            _reg = float(s[reg_i:reg_j])
        reg.add(_reg)

        decay_i = s.find('decay_') + 6
        decay_j = s.find('-', decay_i)
        _decay = float(s[decay_i:decay_j])
        decay.add(_decay)

        files = os.listdir(s_path)
        # files = [f'sample_{i}.pkl' for i in range(1, 51)]
        samples = 0
        success = []
        x0, y0 = [], []
        status = []
        solve_time = []
        for f in files:
            if f.endswith('.pkl'):
                samples += 1
                f_path = pathlib.Path(s_path, f)
                try:
                    with open(f_path, 'rb') as _f:
                        d = pickle.load(_f)
                except:
                    break
                msg = d['solver_result']['msg']
                if 'path' in s and msg == 'MCP_Solved':
                    success.append(True)
                elif 'dgsqp' in s and msg == 'conv_abs_tol':
                    success.append(True)
                else:
                    success.append(False)
                status.append(msg)

                solve_time.append(d['solver_result']['time'])

                initial_condition = d['initial_condition']
                x0.append(np.mean([s.x.x for s in initial_condition]))
                y0.append(np.mean([s.x.y for s in initial_condition]))

        solve_time = np.array(solve_time)
        x0 = np.array(x0)
        y0 = np.array(y0)

        if samples > 0:
            if np.sum(success) >= 1:
                _t = solve_time[np.where(success)[0]]
                _avg = np.mean(_t)
                _max = np.amax(_t)
                _min = np.amin(_t)
            else:
                _avg, _max, _min = np.nan, np.nan, np.nan
            print(f'{s}: {np.sum(success)}/{samples} | {_avg:.2f}, {_max:.2f}, {_min:.2f}')
            _success_rate[(_reg, _decay)] = np.sum(success)/samples
            _solve_time[(_reg, _decay)] = _avg
        else:
            _success_rate[(_reg, _decay)] = np.nan
            _solve_time[(_reg, _decay)] = np.nan

# reg_vals = sorted(reg)
# decay_vals = sorted(decay)
# R, D = np.meshgrid(reg_vals, decay_vals)
# success_rate = np.zeros(R.shape)
# solve_time = np.zeros(R.shape)

# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         if (R[i,j], D[i,j]) in _success_rate.keys():
#             if R[i,j] == 0:
#                 success_rate[:,j] = _success_rate[(R[i,j], D[i,j])]
#                 solve_time[:,j] = _solve_time[(R[i,j], D[i,j])]
#             else:
#                 success_rate[i,j] = _success_rate[(R[i,j], D[i,j])]
#                 solve_time[i,j] = _solve_time[(R[i,j], D[i,j])]

# # Success rate
# fig, ax = plt.subplots(figsize=(15,10))
# ax.set_title(f'Success rate')
# im = ax.imshow(success_rate, vmin=0, vmax=1) 
# for i in range(len(decay_vals)):
#     for j in range(len(reg_vals)):
#         ax.text(j, i, f'${str(success_rate[i,j])}$', 
#                     color='black', 
#                     fontsize=20,
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'))
# ax.set_xticks(np.arange(len(reg_vals)), [f'${v}$' for v in reg_vals])
# ax.set_yticks(np.arange(len(decay_vals)), [f'${v}$' for v in decay_vals])
# ax.set_xlabel('Regularization weight', fontsize=20)
# ax.set_ylabel('Decay rate', fontsize=20)
# ax.tick_params(axis='x', which='major', labelsize=20)
# ax.tick_params(axis='y', which='major', labelsize=20)

# sm = ScalarMappable(None)
# sm.set_clim(0, 1)
# cb = fig.colorbar(sm, ax=ax)
# cb.set_label('success rate', size=20)
# cb.ax.tick_params(labelsize=20)

# fig.savefig(pathlib.Path(out_path, f'regularization_success_{eval_type}.png'), dpi=200, bbox_inches='tight')

# # Solve time
# fig, ax = plt.subplots(figsize=(15,10))
# ax.set_title(f'Solve time')
# im = ax.imshow(solve_time) 
# for i in range(len(decay_vals)):
#     for j in range(len(reg_vals)):
#         ax.text(j, i, f'${solve_time[i,j]:.2f}$', 
#                     color='black', 
#                     fontsize=20,
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'))
# ax.set_xticks(np.arange(len(reg_vals)), [f'${v}$' for v in reg_vals])
# ax.set_yticks(np.arange(len(decay_vals)), [f'${v}$' for v in decay_vals])
# ax.set_xlabel('Regularization weight', fontsize=20)
# ax.set_ylabel('Decay rate', fontsize=20)
# ax.tick_params(axis='x', which='major', labelsize=20)
# ax.tick_params(axis='y', which='major', labelsize=20)

# sm = ScalarMappable(None)
# sm.set_clim(np.nanmin(solve_time), np.nanmax(solve_time))
# cb = fig.colorbar(sm, ax=ax)
# cb.set_label('solve time [s]', size=20)
# cb.ax.tick_params(labelsize=20)

# fig.savefig(pathlib.Path(out_path, f'regularization_solve_time_{eval_type}.png'), dpi=200, bbox_inches='tight')

# plt.show()