#!/usr/bin python3

import numpy as np
import pathlib
import pickle
import os

from DGSQP.tracks.track_lib import get_track

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['text.usetex'] = True

data_dir = 'dgsqp_algames_mc_comp_09-10-2022_19-41-37'
data_path = pathlib.Path(pathlib.Path.home(), f'results/{data_dir}')

sqp_conv_feas, sqp_max_feas, sqp_fail_feas = [], [], [] 
alg_conv_feas, alg_max_feas, alg_fail_feas = [], [], [] 
n_sqp_conv_feas = 0
n_sqp_fail_feas = 0
n_alg_conv_feas = 0
n_alg_fail_feas = 0

k = 0
a = 0

sqp_n_conv, sqp_n_div, sqp_n_max = 0, 0, 0
sqp_iters, sqp_time = [], []

alg_n_conv, alg_n_div, alg_n_max = 0, 0, 0
alg_iters, alg_time = [], []

x0, y0 = [], []
sqp_status, alg_status = [], []

for f in os.listdir(data_path):
    if f.endswith('.pkl'):
        with open(data_path.joinpath(f), 'rb') as f:
            data = pickle.load(f)
        
        joint_init = data['dgsqp']['init']
        x0.append(np.mean([s.x.x for s in joint_init]))
        y0.append(np.mean([s.x.y for s in joint_init]))

        sqp_res, alg_res = data['dgsqp'], data['algames']

        sqp_conv = sqp_res['solve_info']['status']
        sqp_msg = sqp_res['solve_info']['msg']
        sqp_vio = sqp_res['solve_info']['cond']['p_feas']
        if sqp_conv: 
            sqp_n_conv += 1
            sqp_iters.append(sqp_res['solve_info']['num_iters'])
            sqp_time.append(sqp_res['solve_info']['time'])
            if sqp_vio > 0:
                sqp_conv_feas.append(sqp_vio)
            else:
                n_sqp_conv_feas += 1
        if sqp_msg == 'max_it':
            sqp_n_max += 1
            if sqp_vio > 0:
                sqp_max_feas.append(sqp_vio)
            else:
                n_sqp_fail_feas += 1
        elif sqp_msg == 'diverged' or sqp_msg == 'qp_fail':
            sqp_n_div += 1
            if sqp_vio > 0:
                sqp_fail_feas.append(sqp_vio)
            else:
                n_sqp_fail_feas += 1
        sqp_status.append(sqp_conv)
        
        alg_conv = alg_res['solve_info']['status']
        alg_msg = alg_res['solve_info']['msg']
        alg_vio = np.abs(alg_res['solve_info']['cond']['p_feas'])
        if alg_conv: 
            alg_n_conv += 1
            alg_iters.append(alg_res['solve_info']['num_iters'])
            alg_time.append(alg_res['solve_info']['time'])
            if alg_vio > 0:
                alg_conv_feas.append(alg_vio)
            else:
                n_alg_conv_feas += 1
        if alg_msg == 'max_iters' or alg_msg == 'max_it':
            alg_n_max += 1
            if alg_vio > 0:
                alg_max_feas.append(alg_vio)
            else:
                n_alg_fail_feas += 1
        elif alg_msg == 'diverged':
            alg_n_div += 1 
            if alg_vio > 0:
                alg_fail_feas.append(alg_vio)
            else:
                n_alg_fail_feas += 1
        alg_status.append(alg_conv)

        a += (np.mean(sqp_iters)-np.mean(alg_iters))/np.mean(alg_iters)
        k += 1
# print(n_sqp_conv_feas, n_sqp_fail_feas)
# print(n_alg_conv_feas, n_alg_fail_feas)
# print(a/k)
x0 = np.array(x0)
y0 = np.array(y0)

w = 7
print('========================================')
print('          |  SQP  |  ALG  ')
print('Converged |%s|%s' % (f'{sqp_n_conv:3d}'.rjust(w), f'{alg_n_conv:3d}'.rjust(w)))
print('Failed    |%s|%s' % (f'{sqp_n_div:3d}'.rjust(w), f'{alg_n_div:3d}'.rjust(w)))
print('Max       |%s|%s' % (f'{sqp_n_max:3d}'.rjust(w), f'{alg_n_max:3d}'.rjust(w)))
print('Avg iters |%s|%s' % (f'{np.mean(sqp_iters):4.2f}'.rjust(w), f'{np.mean(alg_iters):4.2f}'.rjust(w)))
print('Std iters |%s|%s' % (f'{np.std(sqp_iters):4.2f}'.rjust(w), f'{np.std(alg_iters):4.2f}'.rjust(w)))
print('Avg time  |%s|%s' % (f'{np.mean(sqp_time):4.2f}'.rjust(w), f'{np.mean(alg_time):4.2f}'.rjust(w)))
print('Std time  |%s|%s' % (f'{np.std(sqp_time):4.2f}'.rjust(w), f'{np.std(alg_time):4.2f}'.rjust(w)))

track_obj = get_track('L_track_barc')

fig = plt.figure(figsize=(20,10))
ax_sqp = fig.add_subplot(1,2,1)
track_obj.plot_map(ax_sqp)
sqp_succ = np.where(sqp_status)[0].astype(int)
sqp_fail = np.setdiff1d(np.arange(len(sqp_status)), sqp_succ).astype(int)
ax_sqp.scatter(x0[sqp_succ], y0[sqp_succ], facecolors='none', edgecolors='g', marker='o', s=40)
ax_sqp.scatter(x0[sqp_fail], y0[sqp_fail], c='r', marker='x', s=40)
ax_sqp.set_aspect('equal')
ax_sqp.get_xaxis().set_ticks([])
ax_sqp.get_yaxis().set_ticks([])
# ax_sqp.tick_params(axis='x', labelsize=15)
# ax_sqp.tick_params(axis='y', labelsize=15)

ax_alg = fig.add_subplot(1,2,2)
track_obj.plot_map(ax_alg)
alg_succ = np.where(alg_status)[0].astype(int)
alg_fail = np.setdiff1d(np.arange(len(alg_status)), alg_succ).astype(int)
ax_alg.scatter(x0[alg_succ], y0[alg_succ], facecolors='none', edgecolors='g', marker='o', s=40)
ax_alg.scatter(x0[alg_fail], y0[alg_fail], c='r', marker='x', s=40)
ax_alg.set_aspect('equal')
ax_alg.get_xaxis().set_ticks([])
ax_alg.get_yaxis().set_ticks([])
# ax_alg.tick_params(axis='x', labelsize=15)

fig.subplots_adjust(wspace=0.01)

plt.show()
