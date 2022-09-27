#!/usr/bin python3

import numpy as np
import pathlib
import pickle
import pdb

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['text.usetex'] = True

# Change to location of data files
data_dir = 'dgsqp_algames_mc_ablation_09-11-2022_10-52-33'
data_path = pathlib.Path(pathlib.Path.home(), f'results/{data_dir}')

exp_track_param = [90]
exp_N = [15, 20, 25]

sqp_conv_feas, sqp_max_feas, sqp_fail_feas = [], [], [] 
alg_conv_feas, alg_max_feas, alg_fail_feas = [], [], [] 

for c in exp_track_param:
    for N in exp_N:
        filename = f'data_c_{c}_N_{N}.pkl'
        with open(data_path.joinpath(filename), 'rb') as f:
            data = pickle.load(f)
        
        n_trials = len(data['sqgames_all'])
        sqp_all_res, sqp_none_res = data['sqgames_all'], data['sqgames_none']
        sqp_all_opt, sqp_none_opt = [], []
        
        for i in range(n_trials):
            sqp_all_opt.append(sqp_all_res[i]['solve_info']['cond']['stat'])
            sqp_none_opt.append(sqp_none_res[i]['solve_info']['cond']['stat'])

        # Plot optimality histograms
        font_size = 20
        bins = 10
        fig_opt_hist = plt.figure(figsize=(9, 3))
        ax_opt = fig_opt_hist.add_subplot(1, 1, 1)

        ax_sqp_all_opt = fig_opt_hist.add_subplot(1, 2, 1)
        ax_sqp_all_opt.hist(np.log10(sqp_all_opt), bins=bins, edgecolor='black', linewidth=1.2)
        ax_sqp_all_opt.set_xlim([-4, 2])
        ax_sqp_all_opt.tick_params(axis='x', labelsize=font_size)
        ax_sqp_all_opt.tick_params(axis='y', labelsize=font_size)
        ax_sqp_none_opt = fig_opt_hist.add_subplot(1, 2, 2)
        ax_sqp_none_opt.hist(np.log10(sqp_none_opt), bins=bins, edgecolor='black', linewidth=1.2)
        ax_sqp_none_opt.set_xlim([-4, 2])
        ax_sqp_none_opt.tick_params(axis='x', labelsize=font_size)
        ax_sqp_none_opt.tick_params(axis='y', labelsize=font_size)

        ax_opt.spines['top'].set_color('none')
        ax_opt.spines['bottom'].set_color('none')
        ax_opt.spines['left'].set_color('none')
        ax_opt.spines['right'].set_color('none')
        ax_opt.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax_opt.set_xlabel(r'$\log_{10}$ optimality', fontsize=font_size, labelpad=10)
        plt.tight_layout()

        print(np.median(sqp_all_opt), np.median(sqp_none_opt))
plt.show()