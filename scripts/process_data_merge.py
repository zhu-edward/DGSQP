#!/usr/bin python3

import numpy as np
import pathlib
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['text.usetex'] = True

data_dir = 'dgsqp_algames_mc_merge_09-06-2022_14-15-52'
data_path = pathlib.Path(pathlib.Path.home(), f'results/{data_dir}')

sqp_conv_feas, sqp_max_feas, sqp_fail_feas = [], [], [] 
n_sqp_conv_feas = 0
n_sqp_fail_feas = 0

k = 0
a = 0

sqp_n_conv, sqp_n_div, sqp_n_max = 0, 0, 0
sqp_iters, sqp_time, sqp_opt = [], [], []

for f in os.listdir(data_path):
    if f.endswith('.pkl'):
        with open(data_path.joinpath(f), 'rb') as f:
            data = pickle.load(f)
        
        sqp_res = data['dgsqp']

        sqp_conv = sqp_res['solve_info']['status']
        sqp_msg = sqp_res['solve_info']['msg']
        sqp_vio = sqp_res['solve_info']['cond']['p_feas']
        if sqp_conv: 
            sqp_n_conv += 1
            sqp_iters.append(sqp_res['solve_info']['num_iters'])
            sqp_time.append(sqp_res['solve_info']['time'])
            sqp_opt.append(sqp_res['solve_info']['cond']['stat'])
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

w = 7
print('========================================')
print('          |  SQP  ')
print('Converged |%s' % (f'{sqp_n_conv:3d}'.rjust(w)))
print('Failed    |%s' % (f'{sqp_n_div:3d}'.rjust(w)))
print('Max       |%s' % (f'{sqp_n_max:3d}'.rjust(w)))
print('Avg iters |%s' % (f'{np.mean(sqp_iters):4.2f}'.rjust(w)))
print('Std iters |%s' % (f'{np.std(sqp_iters):4.2f}'.rjust(w)))
print('Avg time  |%s' % (f'{np.mean(sqp_time):4.2f}'.rjust(w)))
print('Std time  |%s' % (f'{np.std(sqp_time):4.2f}'.rjust(w)))