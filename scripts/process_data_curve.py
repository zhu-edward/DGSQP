#!/usr/bin python3

import numpy as np
import pathlib
import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['text.usetex'] = True

data_dir = 'dgsqp_algames_mc_curve_09-10-2022_17-44-49'
data_path = pathlib.Path(pathlib.Path.home(), f'results/data_for_paper/{data_dir}')

exp_track_param = [45, 75, 90]
exp_N = [10, 15, 20, 25]

sqp_conv_feas, sqp_max_feas, sqp_fail_feas = [], [], [] 
alg_conv_feas, alg_max_feas, alg_fail_feas = [], [], [] 
n_sqp_conv_feas = 0
n_sqp_fail_feas = 0
n_alg_conv_feas = 0
n_alg_fail_feas = 0

k = 0
a = 0

for c in exp_track_param:
    for N in exp_N:
        filename = f'data_c_{c}_N_{N}.pkl'
        with open(data_path.joinpath(filename), 'rb') as f:
            data = pickle.load(f)
        
        n_trials = len(data['sqgames'])
        sqp_res, alg_res = data['sqgames'], data['algames']

        sqp_n_conv, sqp_n_div, sqp_n_max = 0, 0, 0
        sqp_iters, sqp_solves, sqp_time = [], [], []
        
        alg_n_conv, alg_n_div, alg_n_max = 0, 0, 0
        alg_iters, alg_solves, alg_time = [], [], []
        
        for i in range(n_trials):
            sqp_conv = sqp_res[i]['solve_info']['status']
            sqp_msg = sqp_res[i]['solve_info']['msg']
            sqp_vio = sqp_res[i]['solve_info']['cond']['p_feas']
            sqp_it = sqp_res[i]['solve_info']['num_iters']
            if sqp_conv: 
                sqp_n_conv += 1
                qp_solves = np.sum([d['qp_solves'] for d in sqp_res[i]['solve_info']['iter_data']])
                sqp_solves.append(qp_solves)
                sqp_iters.append(sqp_it)
                sqp_time.append(sqp_res[i]['solve_info']['time'])
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
            
            alg_conv = alg_res[i]['solve_info']['status']
            alg_msg = alg_res[i]['solve_info']['msg']
            alg_vio = np.abs(alg_res[i]['solve_info']['cond']['p_feas'])
            alg_it = alg_res[i]['solve_info']['num_iters']
            if alg_conv: 
                alg_n_conv += 1
                newton_solves = np.sum([d['newton_solves'] for d in alg_res[i]['solve_info']['iter_data']])
                alg_solves.append(newton_solves)
                alg_iters.append(alg_it)
                alg_time.append(alg_res[i]['solve_info']['time'])
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

        w = 7
        print('========================================')
        print(f'Track parameter: {c}, N: {N}')
        print('           |  SQP  |  ALG  ')
        print('Converged  |%s|%s' % (f'{sqp_n_conv:3d}'.rjust(w), f'{alg_n_conv:3d}'.rjust(w)))
        print('Failed     |%s|%s' % (f'{sqp_n_div:3d}'.rjust(w), f'{alg_n_div:3d}'.rjust(w)))
        print('Max        |%s|%s' % (f'{sqp_n_max:3d}'.rjust(w), f'{alg_n_max:3d}'.rjust(w)))
        print('Avg iters  |%s|%s' % (f'{np.mean(sqp_iters):4.2f}'.rjust(w), f'{np.mean(alg_iters):4.2f}'.rjust(w)))
        print('Std iters  |%s|%s' % (f'{np.std(sqp_iters):4.2f}'.rjust(w), f'{np.std(alg_iters):4.2f}'.rjust(w)))
        print('Avg solves |%s|%s' % (f'{np.mean(sqp_solves):4.2f}'.rjust(w), f'{np.mean(alg_solves):4.2f}'.rjust(w)))
        print('Std solves |%s|%s' % (f'{np.std(sqp_solves):4.2f}'.rjust(w), f'{np.std(alg_solves):4.2f}'.rjust(w)))
        print('Avg time   |%s|%s' % (f'{np.mean(sqp_time):4.2f}'.rjust(w), f'{np.mean(alg_time):4.2f}'.rjust(w)))
        print('Std time   |%s|%s' % (f'{np.std(sqp_time):4.2f}'.rjust(w), f'{np.std(alg_time):4.2f}'.rjust(w)))

        a += (np.mean(sqp_solves)-np.mean(alg_solves))/np.mean(alg_solves)
        k += 1
# print(n_sqp_conv_feas, n_sqp_fail_feas)
# print(n_alg_conv_feas, n_alg_fail_feas)
# print(a/k)

# Plot constraint violation and complementarity histograms
font_size = 20
bins = 10

fig_conv_feas_hist = plt.figure(figsize=(9, 3))
ax_conv_feas = fig_conv_feas_hist.add_subplot(1, 1, 1)

ax_sqp_conv_feas = fig_conv_feas_hist.add_subplot(1, 2, 1)
ax_sqp_conv_feas.hist(np.log10(sqp_conv_feas), bins=bins, edgecolor='black', linewidth=1.2)
ax_sqp_conv_feas.set_xlim([-16, 0])
ax_sqp_conv_feas.tick_params(axis='x', labelsize=font_size)
ax_sqp_conv_feas.tick_params(axis='y', labelsize=font_size)
ax_alg_conv_feas = fig_conv_feas_hist.add_subplot(1, 2, 2)
ax_alg_conv_feas.hist(np.log10(alg_conv_feas), bins=bins, edgecolor='black', linewidth=1.2)
ax_alg_conv_feas.set_xlim([-16, 0])
ax_alg_conv_feas.tick_params(axis='x', labelsize=font_size)
ax_alg_conv_feas.tick_params(axis='y', labelsize=font_size)

ax_conv_feas.spines['top'].set_color('none')
ax_conv_feas.spines['bottom'].set_color('none')
ax_conv_feas.spines['left'].set_color('none')
ax_conv_feas.spines['right'].set_color('none')
ax_conv_feas.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
ax_conv_feas.set_xlabel(r'$\log_{10}$ constraint violation', fontsize=font_size, labelpad=10)
plt.tight_layout()

fig_fail_feas_hist = plt.figure(figsize=(9, 3))
ax_fail_feas = fig_fail_feas_hist.add_subplot(1, 1, 1)

ax_sqp_fail_feas = fig_fail_feas_hist.add_subplot(1, 2, 1)
ax_sqp_fail_feas.hist(np.log10(sqp_max_feas+sqp_fail_feas), bins=bins, edgecolor='black', linewidth=1.2)
ax_sqp_fail_feas.set_xlim([-10, 10])
ax_sqp_fail_feas.tick_params(axis='x', labelsize=font_size)
ax_sqp_fail_feas.tick_params(axis='y', labelsize=font_size)
ax_sqp_fail_feas.plot([-3, -3], ax_sqp_fail_feas.get_ylim(), 'k--')
ax_alg_fail_feas = fig_fail_feas_hist.add_subplot(1, 2, 2)
ax_alg_fail_feas.hist(np.log10(alg_max_feas+alg_fail_feas), bins=bins, edgecolor='black', linewidth=1.2)
ax_alg_fail_feas.set_xlim([-10, 10])
ax_alg_fail_feas.tick_params(axis='x', labelsize=font_size)
ax_alg_fail_feas.tick_params(axis='y', labelsize=font_size)
ax_alg_fail_feas.plot([-3, -3], ax_alg_fail_feas.get_ylim(), 'k--')

ax_fail_feas.spines['top'].set_color('none')
ax_fail_feas.spines['bottom'].set_color('none')
ax_fail_feas.spines['left'].set_color('none')
ax_fail_feas.spines['right'].set_color('none')
ax_fail_feas.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
ax_fail_feas.set_xlabel(r'$\log_{10}$ constraint violation', fontsize=font_size, labelpad=10)
plt.tight_layout()

plt.show()