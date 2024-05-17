#!/usr/bin python3

import numpy as np
import scipy as sp
import pathlib
import pickle
import pdb
import os

from DGSQP.tracks.track_lib import get_track
from DGSQP.dynamics.dynamics_models import CasadiKinematicBicycleCombined, CasadiKinematicBicycleProgressAugmented, CasadiDynamicBicycleCombined, CasadiDynamicBicycleProgressAugmented, CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.dynamics.model_types import DynamicBicycleConfig, KinematicBicycleConfig, MultiAgentModelConfig

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['text.usetex'] = True

from globals import TRACK

track_obj = get_track(TRACK)

base_dir = pathlib.Path(pathlib.Path.home(), 'results/tro_data_v2/')

study_dir = 'comparison_study_barc_dynamic_paper_2024-04-25_16-57-29'

dt = 0.1
discretization_method = 'rk4'
M = 10
# discretization_method='rk4'
car1_dynamics_config = DynamicBicycleConfig(dt=dt,
                                                model_name='dynamic_bicycle',
                                                noise=False,
                                                discretization_method=discretization_method,
                                                simple_slip=False,
                                                tire_model='pacejka',
                                                mass=2.2187,
                                                yaw_inertia=0.02723,
                                                wheel_friction=0.9,
                                                pacejka_b_front=5.0,
                                                pacejka_b_rear=5.0,
                                                pacejka_c_front=2.28,
                                                pacejka_c_rear=2.28,
                                                M=M)
car1_dyn_model = CasadiDynamicBicycleCombined(0, car1_dynamics_config, track=track_obj)
car1_pa_dyn_model = CasadiDynamicBicycleProgressAugmented(0, car1_dynamics_config, track=track_obj)

car2_dynamics_config = DynamicBicycleConfig(dt=dt,
                                                model_name='dynamic_bicycle',
                                                noise=False,
                                                discretization_method=discretization_method,
                                                simple_slip=False,
                                                tire_model='pacejka',
                                                mass=2.2187,
                                                yaw_inertia=0.02723,
                                                wheel_friction=0.9,
                                                pacejka_b_front=5.0,
                                                pacejka_b_rear=5.0,
                                                pacejka_c_front=2.28,
                                                pacejka_c_rear=2.28,
                                                M=M)
car2_dyn_model = CasadiDynamicBicycleCombined(0, car2_dynamics_config, track=track_obj)
car2_pa_dyn_model = CasadiDynamicBicycleProgressAugmented(0, car2_dynamics_config, track=track_obj)

joint_config = MultiAgentModelConfig(dt=dt,
                                    discretization_method=discretization_method,
                                    use_mx=False,
                                    code_gen=False,
                                    verbose=False,
                                    compute_hessians=False,
                                    M=M)
joint_model = CasadiDecoupledMultiAgentDynamicsModel(0, [car1_dyn_model, car2_dyn_model], joint_config)
joint_pa_model = CasadiDecoupledMultiAgentDynamicsModel(0, [car1_pa_dyn_model, car2_pa_dyn_model], joint_config)

def rollout_gne(f, q0, u):
    q = [q0]
    for k in range(u.shape[0]):
        q.append(np.array(f.fd(q[-1], u[k])).squeeze())
    return np.array(q)

N = 15

solver_1 = 'approximate-dgsqp-once'
# solver_1 = 'exact-path'

if 'dgsqp' in solver_1:
    data_dir_1 = f'N{N}-dynamic-{solver_1}-stat_l1-armijo-adaptive-nms'
else:
    data_dir_1 = f'N{N}-dynamic-{solver_1}-nms'

solver_2 = 'exact-dgsqp'

if 'dgsqp' in solver_2:
    data_dir_2 = f'N{N}-dynamic-{solver_2}-stat_l1-armijo-adaptive-nms'
else:
    data_dir_2 = f'N{N}-dynamic-{solver_2}-nms'

M = 3

show_plots = False

all_success = 0

results = {solver_1: dict(success=[], status=[], fail_counts={}, num_iters=[], solve_time=[], solution=[], initial_condition=[]),
           solver_2: dict(success=[], status=[], fail_counts={}, num_iters=[], solve_time=[], solution=[], initial_condition=[])}
_dist = []
_idxs = []

data_path_1 = pathlib.Path(base_dir, study_dir, data_dir_1)
data_path_2 = pathlib.Path(base_dir, study_dir, data_dir_2)

j = 0
while True:
    filename = f'sample_{j+1}.pkl'

    file_path_1 = data_path_1.joinpath(filename)
    file_path_2 = data_path_2.joinpath(filename)

    exist_1 = os.path.exists(file_path_1)
    exist_2 = os.path.exists(file_path_2)
    if not exist_1 and not exist_2:
        break
    elif not exist_1 or not exist_2:
        j += 1
        continue

    with open(file_path_1, 'rb') as f:
        data_1 = pickle.load(f)
    with open(file_path_2, 'rb') as f:
        data_2 = pickle.load(f)

    results[solver_1]['initial_condition'].append(data_1['initial_condition'])
    results[solver_2]['initial_condition'].append(data_2['initial_condition'])

    if 'dgsqp' in solver_1:
        if data_1['solver_result']['msg'] == 'conv_abs_tol':
            results[solver_1]['success'].append(True)
        else:
            results[solver_1]['success'].append(False)
    else:
        if data_1['solver_result']['status']:
            results[solver_1]['success'].append(True)
        else:
            results[solver_1]['success'].append(False)
    
    if 'dgsqp' in solver_2:
        if data_2['solver_result']['msg'] == 'conv_abs_tol':
            results[solver_2]['success'].append(True)
        else:
            results[solver_2]['success'].append(False)
    else:
        if data_1['solver_result']['status']:
            results[solver_2]['success'].append(True)
        else:
            results[solver_2]['success'].append(False)

    solution_1 = data_1['solver_result']['primal_sol'].reshape((2*N,-1))
    solution_1 = np.hstack((solution_1[:N], solution_1[N:]))
    if 'approximate' in solver_1:
        solution_1 = solution_1[:,[0,1,3,4]]
    results[solver_1]['solution'].append(solution_1)

    solution_2 = data_2['solver_result']['primal_sol'].reshape((2*N,-1))
    solution_2 = np.hstack((solution_2[:N], solution_2[N:]))
    if 'approximate' in solver_2:
        solution_2 = solution_2[:,[0,1,3,4]]
    results[solver_2]['solution'].append(solution_2)

    if results[solver_1]['success'][-1] and results[solver_2]['success'][-1]:
        _dist.append(np.linalg.norm(np.divide(results[solver_1]['solution'][-1]-results[solver_2]['solution'][-1], np.array([2,0.436,2,0.436])), ord='fro')/N)
        _idxs.append(j)
        all_success += 1

    j += 1

print(f'{solver_1} vs. {solver_2}')
print(f'MSE min: {np.amin(_dist)}, mean: {np.mean(_dist)},  median: {np.median(_dist)}, max: {np.amax(_dist)}')

w = 12
# print('========================================')
solver_names = results.keys()
header    = ['         '] + [n.rjust(w) for n in solver_names]
converged = ['Converged'] + [f"{np.sum(results[n]['success']):3d}".rjust(w) for n in solver_names]

print(' | '.join(header))
print(' | '.join(converged))

out_path = pathlib.Path(base_dir, study_dir, 'out', f'{solver_1}_{solver_2}')
if not os.path.exists(out_path):
    os.makedirs(out_path)

legend = False

bins = np.linspace(0, 0.3, 13)

fig_dist = plt.figure(figsize=(8,2))
ax_dist = fig_dist.gca()
ax_dist.hist(_dist, bins, edgecolor='black', linewidth=1.2)
y_lim = ax_dist.get_ylim()
ax_dist.plot([np.mean(_dist), np.mean(_dist)], y_lim, 'k--', linewidth=2)
ax_dist.set_xlim([-0.012, 0.3])
ax_dist.set_ylim(y_lim)
ax_dist.set_xlabel('Normalized MSE', fontsize=15)
ax_dist.tick_params(axis='x', labelsize=15)
ax_dist.tick_params(axis='y', labelsize=15)
ax_dist.set_title(f'{solver_1} - {solver_2} | N = {N} | samples: {all_success}')
fig_dist.savefig(pathlib.Path(out_path, f'distance_histogram.png'), dpi=200, bbox_inches='tight')

if 'approximate' in solver_1:
    f_1 = joint_pa_model
    xy_idx_1 = [[3, 4], [10, 11]]
else:
    f_1 = joint_model
    xy_idx_1 = [[0, 1], [8, 9]]
    se_idx_1 = [[6, 7], [14, 15]]
if 'approximate' in solver_2:
    f_2 = joint_pa_model
    xy_idx_2 = [[3, 4], [10, 11]]
else:
    f_2 = joint_model
    xy_idx_2 = [[0, 1], [8, 9]]
    se_idx_2 = [[6, 7], [14, 15]]

line_width = 2
marker_size = 7
for j in range(M):
    idx_best = _idxs[np.argsort(_dist)[j]]

    x0_1 = f_1.state2q(results[solver_1]['initial_condition'][idx_best])
    u_gne_1 = results[solver_1]['solution'][idx_best]
    if 'approximate' in solver_1:
        u_gne_1 = np.hstack((u_gne_1[:,:2], np.zeros((u_gne_1.shape[0],1)), u_gne_1[:,2:], np.zeros((u_gne_1.shape[0],1))))
    x_gne_1 = rollout_gne(f_1, x0_1, u_gne_1)
    if 'approximate' not in solver_1:
        for k in range(x_gne_1.shape[0]):
            _x1, _y1, _ = track_obj.local_to_global((x_gne_1[k,se_idx_1[0][0]], x_gne_1[k,se_idx_1[0][1]], 0))
            _x2, _y2, _ = track_obj.local_to_global((x_gne_1[k,se_idx_1[1][0]], x_gne_1[k,se_idx_1[1][1]], 0))
            x_gne_1[k,xy_idx_1[0]] = np.array([_x1, _y1])
            x_gne_1[k,xy_idx_1[1]] = np.array([_x2, _y2])

    # print(solver_1)
    # print(u_gne_1)

    x0_2 = f_2.state2q(results[solver_2]['initial_condition'][idx_best])
    u_gne_2 = results[solver_2]['solution'][idx_best]
    if 'approximate' in solver_2:
        u_gne_2 = np.hstack((u_gne_2[:,:2], np.zeros((u_gne_2.shape[0],1)), u_gne_2[:,2:], np.zeros((u_gne_2.shape[0],1))))
    x_gne_2 = rollout_gne(f_2, x0_2, u_gne_2)
    if 'approximate' not in solver_2:
        for k in range(x_gne_2.shape[0]):
            _x1, _y1, _ = track_obj.local_to_global((x_gne_2[k,se_idx_2[0][0]], x_gne_2[k,se_idx_2[0][1]], 0))
            _x2, _y2, _ = track_obj.local_to_global((x_gne_2[k,se_idx_2[1][0]], x_gne_2[k,se_idx_2[1][1]], 0))
            x_gne_2[k,xy_idx_2[0]] = np.array([_x1, _y1])
            x_gne_2[k,xy_idx_2[1]] = np.array([_x2, _y2])

    # print(solver_2)
    # print(u_gne_2)

    fig_best = plt.figure(figsize=(10,10))
    ax_best = fig_best.gca()
    track_obj.plot_map(ax_best)
    ax_best.plot(x_gne_1[:,xy_idx_1[0][0]], x_gne_1[:,xy_idx_1[0][1]], 'bo--', linewidth=line_width, markersize=marker_size, label=solver_1)
    ax_best.plot(x_gne_1[:,xy_idx_1[1][0]], x_gne_1[:,xy_idx_1[1][1]], 'go--', linewidth=line_width, markersize=marker_size, label=solver_1)
    ax_best.plot(x_gne_2[:,xy_idx_2[0][0]], x_gne_2[:,xy_idx_2[0][1]], 'bs-', linewidth=line_width, markersize=marker_size, label=solver_2)
    ax_best.plot(x_gne_2[:,xy_idx_2[1][0]], x_gne_2[:,xy_idx_2[1][1]], 'gs-', linewidth=line_width, markersize=marker_size, label=solver_2)
    ax_best.set_title(f'best diff {j}')
    ax_best.set_aspect('equal')
    ax_best.get_xaxis().set_ticks([])
    ax_best.get_yaxis().set_ticks([])
    if legend:
        ax_best.legend(fontsize=20)
    fig_best.savefig(pathlib.Path(out_path, f'best_diff_{j}.png'), dpi=200, bbox_inches='tight')

for j in range(M):
    idx_mean = _idxs[np.argsort(np.abs(_dist-np.mean(_dist)))[j]]

    x0_1 = f_1.state2q(results[solver_1]['initial_condition'][idx_mean])
    u_gne_1 = results[solver_1]['solution'][idx_mean]
    if 'approximate' in solver_1:
        u_gne_1 = np.hstack((u_gne_1[:,:2], np.zeros((u_gne_1.shape[0],1)), u_gne_1[:,2:], np.zeros((u_gne_1.shape[0],1))))
    x_gne_1 = rollout_gne(f_1, x0_1, u_gne_1)
    if 'approximate' not in solver_1:
        for k in range(x_gne_1.shape[0]):
            _x1, _y1, _ = track_obj.local_to_global((x_gne_1[k,se_idx_1[0][0]], x_gne_1[k,se_idx_1[0][1]], 0))
            _x2, _y2, _ = track_obj.local_to_global((x_gne_1[k,se_idx_1[1][0]], x_gne_1[k,se_idx_1[1][1]], 0))
            x_gne_1[k,xy_idx_1[0]] = np.array([_x1, _y1])
            x_gne_1[k,xy_idx_1[1]] = np.array([_x2, _y2])

    x0_2 = f_2.state2q(results[solver_2]['initial_condition'][idx_mean])
    u_gne_2 = results[solver_2]['solution'][idx_mean]
    if 'approximate' in solver_2:
        u_gne_2 = np.hstack((u_gne_2[:,:2], np.zeros((u_gne_2.shape[0],1)), u_gne_2[:,2:], np.zeros((u_gne_2.shape[0],1))))
    x_gne_2 = rollout_gne(f_2, x0_2, u_gne_2)
    if 'approximate' not in solver_2:
        for k in range(x_gne_2.shape[0]):
            _x1, _y1, _ = track_obj.local_to_global((x_gne_2[k,se_idx_2[0][0]], x_gne_2[k,se_idx_2[0][1]], 0))
            _x2, _y2, _ = track_obj.local_to_global((x_gne_2[k,se_idx_2[1][0]], x_gne_2[k,se_idx_2[1][1]], 0))
            x_gne_2[k,xy_idx_2[0]] = np.array([_x1, _y1])
            x_gne_2[k,xy_idx_2[1]] = np.array([_x2, _y2])

    fig_mean = plt.figure(figsize=(10,10))
    ax_mean = fig_mean.gca()
    track_obj.plot_map(ax_mean)
    ax_mean.plot(x_gne_1[:,xy_idx_1[0][0]], x_gne_1[:,xy_idx_1[0][1]], 'bo--', linewidth=line_width, markersize=marker_size, label=solver_1)
    ax_mean.plot(x_gne_1[:,xy_idx_1[1][0]], x_gne_1[:,xy_idx_1[1][1]], 'go--', linewidth=line_width, markersize=marker_size, label=solver_1)
    ax_mean.plot(x_gne_2[:,xy_idx_2[0][0]], x_gne_2[:,xy_idx_2[0][1]], 'bs-', linewidth=line_width, markersize=marker_size, label=solver_2)
    ax_mean.plot(x_gne_2[:,xy_idx_2[1][0]], x_gne_2[:,xy_idx_2[1][1]], 'gs-', linewidth=line_width, markersize=marker_size, label=solver_2)
    ax_mean.set_title(f'mean diff {j}')
    ax_mean.set_aspect('equal')
    ax_mean.get_xaxis().set_ticks([])
    ax_mean.get_yaxis().set_ticks([])
    if legend:
        ax_mean.legend(fontsize=20)
    fig_mean.savefig(pathlib.Path(out_path, f'mean_diff_{j}.png'), dpi=200, bbox_inches='tight')

pdb.set_trace()
for j in range(M):
    idx_worst = _idxs[np.argsort(_dist)[-(j+1)]]

    x0_1 = f_1.state2q(results[solver_1]['initial_condition'][idx_worst])
    u_gne_1 = results[solver_1]['solution'][idx_worst]
    if 'approximate' in solver_1:
        u_gne_1 = np.hstack((u_gne_1[:,:2], np.zeros((u_gne_1.shape[0],1)), u_gne_1[:,2:], np.zeros((u_gne_1.shape[0],1))))
    x_gne_1 = rollout_gne(f_1, x0_1, u_gne_1)
    if 'approximate' not in solver_1:
        for k in range(x_gne_1.shape[0]):
            _x1, _y1, _ = track_obj.local_to_global((x_gne_1[k,se_idx_1[0][0]], x_gne_1[k,se_idx_1[0][1]], 0))
            _x2, _y2, _ = track_obj.local_to_global((x_gne_1[k,se_idx_1[1][0]], x_gne_1[k,se_idx_1[1][1]], 0))
            x_gne_1[k,xy_idx_1[0]] = np.array([_x1, _y1])
            x_gne_1[k,xy_idx_1[1]] = np.array([_x2, _y2])

    x0_2 = f_2.state2q(results[solver_2]['initial_condition'][idx_worst])
    u_gne_2 = results[solver_2]['solution'][idx_worst]
    if 'approximate' in solver_2:
        u_gne_2 = np.hstack((u_gne_2[:,:2], np.zeros((u_gne_2.shape[0],1)), u_gne_2[:,2:], np.zeros((u_gne_2.shape[0],1))))
    x_gne_2 = rollout_gne(f_2, x0_2, u_gne_2)
    if 'approximate' not in solver_2:
        for k in range(x_gne_2.shape[0]):
            _x1, _y1, _ = track_obj.local_to_global((x_gne_2[k,se_idx_2[0][0]], x_gne_2[k,se_idx_2[0][1]], 0))
            _x2, _y2, _ = track_obj.local_to_global((x_gne_2[k,se_idx_2[1][0]], x_gne_2[k,se_idx_2[1][1]], 0))
            x_gne_2[k,xy_idx_2[0]] = np.array([_x1, _y1])
            x_gne_2[k,xy_idx_2[1]] = np.array([_x2, _y2])

    fig_worst = plt.figure(figsize=(10,10))
    ax_worst = fig_worst.gca()
    track_obj.plot_map(ax_worst)
    ax_worst.plot(x_gne_1[:,xy_idx_1[0][0]], x_gne_1[:,xy_idx_1[0][1]], 'bo--', linewidth=line_width, markersize=marker_size, label=solver_1)
    ax_worst.plot(x_gne_1[:,xy_idx_1[1][0]], x_gne_1[:,xy_idx_1[1][1]], 'go--', linewidth=line_width, markersize=marker_size, label=solver_1)
    ax_worst.plot(x_gne_2[:,xy_idx_2[0][0]], x_gne_2[:,xy_idx_2[0][1]], 'bs-', linewidth=line_width, markersize=marker_size, label=solver_2)
    ax_worst.plot(x_gne_2[:,xy_idx_2[1][0]], x_gne_2[:,xy_idx_2[1][1]], 'gs-', linewidth=line_width, markersize=marker_size, label=solver_2)
    ax_worst.set_title(f'worst diff {j}')
    ax_worst.set_aspect('equal')
    ax_worst.get_xaxis().set_ticks([])
    ax_worst.get_yaxis().set_ticks([])
    if legend:
        ax_worst.legend(fontsize=20)
    fig_worst.savefig(pathlib.Path(out_path, f'worst_diff_{j}.png'), dpi=200, bbox_inches='tight')

if show_plots:
    plt.show()

pdb.set_trace()