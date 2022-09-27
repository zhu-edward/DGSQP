#!/usr/bin/env python3

from DGSQP.tracks.track_lib import get_track

import pickle
import pathlib
import pdb
import os
import copy
from collections import deque

import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FFMpegWriter

import matplotlib
matplotlib.use('TkAgg')

def process_raceline(path, scaling, L):
    f = np.load(path)
    raceline_mat = np.vstack((f['x'], f['y'], f['v_long']/scaling, f['v_tran']/scaling, f['psidot']/scaling, f['e_psi'], f['s'], f['e_y'])).T
    T = f['t']*scaling

    raceline_mat2 = copy.copy(raceline_mat)
    raceline_mat2[:,6] += L
    T2 = copy.copy(T)
    T2 += T[-1]
    raceline_two_laps = np.vstack((raceline_mat, raceline_mat2[1:]))
    T_two_laps = np.append(T, T2[1:])
    t_sym = ca.MX.sym('t', 1)
    raceline_interp = []
    for i in range(raceline_mat.shape[1]):
        raceline_interp.append(ca.interpolant(f'x{i}', 'linear', [T_two_laps], raceline_two_laps[:,i]))
    raceline = ca.Function('car1_raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
    s2t = ca.interpolant('car1_s2t', 'linear', [raceline_two_laps[:,6]], T_two_laps)

    return raceline, s2t, raceline_mat

# data_dir = 'comp_09-22-2022_14-58-54'
# data_dir = 'comp_block' # Blocking
# data_dir = 'comp_overtake' # Overtake
data_dir = 'comp_block_fail'
# data_dir = 'comp_overtake_fail'
data_path = pathlib.Path(pathlib.Path.home(), f'results/{data_dir}/data.pkl')

with open(data_path, 'rb') as f:
    data = pickle.load(f)

car1_data = data['car1']
car2_data = data['car2']
game_data = data['game']

track_name = 'L_track_barc'
track_obj = get_track(track_name)
L = track_obj.track_length

car1_raceline_path = os.path.join(os.path.expanduser('~/barc_data'), track_name + '_raceline_1.npz')
car1_raceline, car1_s2t, car1_raceline_mat = process_raceline(car1_raceline_path, 1.7, L)

car2_raceline_path = os.path.join(os.path.expanduser('~/barc_data'), track_name + '_raceline_2.npz')
car2_raceline, car2_s2t, car2_raceline_mat = process_raceline(car2_raceline_path, 1.9, L)

VL = 0.37
VW = 0.195

fig = plt.figure(figsize=(20,10))
ax_xy = fig.add_subplot(1,2,1)
ax_v = fig.add_subplot(3,2,2)
ax_a = fig.add_subplot(3,2,4)
ax_s = fig.add_subplot(3,2,6)
track_obj.plot_map(ax_xy)
ax_xy.plot(car1_raceline_mat[:,0], car1_raceline_mat[:,1], 'b', linewidth=5, alpha=0.2)
ax_xy.plot(car2_raceline_mat[:,0], car2_raceline_mat[:,1], 'g', linewidth=5, alpha=0.2)
# ax_xy.plot(x_ref, y_ref, 'r')
ax_xy.set_aspect('equal')
ax_xy.set_xlabel('X [m]', fontsize=15)
ax_xy.set_ylabel('Y [m]', fontsize=15)
ax_v.set_ylabel('vel [m/s]', fontsize=15)
ax_v.xaxis.set_ticklabels([])
ax_a.set_ylabel('accel [m/s^2]', fontsize=15)
ax_a.xaxis.set_ticklabels([])
ax_s.set_ylabel('steer [rad]', fontsize=15)
ax_s.set_xlabel('time [s]', fontsize=15)

car1_l_pred = ax_xy.plot([], [], 'b-o', markersize=4)[0]
car1_l_ref = ax_xy.plot([], [], 'bs', markersize=7, markerfacecolor='None')[0]
# car1_l_la = ax_xy.plot([], [], 'bs', markersize=4, markerfacecolor='None')[0]
car1_l_v = ax_v.plot([], [], '-bo')[0]
car1_l_a = ax_a.plot([], [], '-bo')[0]
car1_l_s = ax_s.plot([], [], '-bo')[0]

car2_l_pred = ax_xy.plot([], [], 'g-o', markersize=4)[0]
car2_l_ref = ax_xy.plot([], [], 'gs', markersize=7, markerfacecolor='None')[0]
# car2_l_la = ax_xy.plot([], [], 'gs', markersize=4, markerfacecolor='None')[0]
car2_l_v = ax_v.plot([], [], '-go')[0]
car2_l_a = ax_a.plot([], [], '-go')[0]
car2_l_s = ax_s.plot([], [], '-go')[0]

t_buff = deque([], maxlen=60)
car1_v_buff = deque([], maxlen=60)
car1_a_buff = deque([], maxlen=60)
car1_s_buff = deque([], maxlen=60)
car2_a_buff = deque([], maxlen=60)
car2_v_buff = deque([], maxlen=60)
car2_s_buff = deque([], maxlen=60)

car1_rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='b', alpha=0.5)
car2_rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='g', alpha=0.5)
ax_xy.add_patch(car1_rect)
ax_xy.add_patch(car2_rect)

b_left = car1_data[0]['state'].x.x - VL/2
b_bot  = car1_data[0]['state'].x.y - VW/2
r = Affine2D().rotate_around(car1_data[0]['state'].x.x, car1_data[0]['state'].x.y, car1_data[0]['state'].e.psi) + ax_xy.transData
car1_rect.set_xy((b_left,b_bot))
car1_rect.set_transform(r)

b_left = car2_data[0]['state'].x.x - VL/2
b_bot  = car2_data[0]['state'].x.y - VW/2
r = Affine2D().rotate_around(car2_data[0]['state'].x.x, car2_data[0]['state'].x.y, car2_data[0]['state'].e.psi) + ax_xy.transData
car2_rect.set_xy((b_left,b_bot))
car2_rect.set_transform(r)

# plt.show()

fps = 20
T = car1_data[-1]['t'] - car1_data[0]['t']
t_span = np.linspace(car1_data[0]['t'], car1_data[-1]['t'], int(T*fps+1))

t_pred = [d['pred'].t for d in car1_data]
t_game = [d['t'] for d in game_data]
t_data = [d['t'] for d in car1_data]

car1_x = ca.interpolant('car1_x', 'linear', [t_data], [d['state'].x.x for d in car1_data])
car1_y = ca.interpolant('car1_y', 'linear', [t_data], [d['state'].x.y for d in car1_data])
car1_p = ca.interpolant('car1_p', 'linear', [t_data], [d['state'].e.psi for d in car1_data])
car1_v = ca.interpolant('car1_v', 'linear', [t_data], [d['state'].v.v_long for d in car1_data])
car1_a = ca.interpolant('car1_a', 'linear', [t_data], [d['state'].u.u_a for d in car1_data])
car1_s = ca.interpolant('car1_s', 'linear', [t_data], [d['state'].u.u_steer for d in car1_data])

car2_x = ca.interpolant('car2_x', 'linear', [t_data], [d['state'].x.x for d in car2_data])
car2_y = ca.interpolant('car2_y', 'linear', [t_data], [d['state'].x.y for d in car2_data])
car2_p = ca.interpolant('car2_p', 'linear', [t_data], [d['state'].e.psi for d in car2_data])
car2_v = ca.interpolant('car2_v', 'linear', [t_data], [d['state'].v.v_long for d in car2_data])
car2_a = ca.interpolant('car2_a', 'linear', [t_data], [d['state'].u.u_a for d in car2_data])
car2_s = ca.interpolant('car2_s', 'linear', [t_data], [d['state'].u.u_steer for d in car2_data])

writer = FFMpegWriter(fps=fps)
video_path = pathlib.Path(pathlib.Path.home(), f'{data_dir}.mp4')
with writer.saving(fig, video_path, 100):
    for t in t_span:
        writer.grab_frame()

        b_left = float(car1_x(t)) - VL/2
        b_bot  = float(car1_y(t)) - VW/2
        r = Affine2D().rotate_around(float(car1_x(t)), float(car1_y(t)), float(car1_p(t))) + ax_xy.transData
        car1_rect.set_xy((b_left,b_bot))
        car1_rect.set_transform(r)

        b_left = float(car2_x(t)) - VL/2
        b_bot  = float(car2_y(t)) - VW/2
        r = Affine2D().rotate_around(float(car2_x(t)), float(car2_y(t)), float(car2_p(t))) + ax_xy.transData
        car2_rect.set_xy((b_left,b_bot))
        car2_rect.set_transform(r)

        t_buff.append(t)
        car1_v_buff.append(float(car1_v(t)))
        car1_a_buff.append(float(car1_a(t)))
        car1_s_buff.append(float(car1_s(t)))
        car2_v_buff.append(float(car2_v(t)))
        car2_a_buff.append(float(car2_a(t)))
        car2_s_buff.append(float(car2_s(t)))

        car1_l_v.set_data(t_buff, car1_v_buff)
        car1_l_a.set_data(t_buff, car1_a_buff)
        car1_l_s.set_data(t_buff, car1_s_buff)

        car2_l_v.set_data(t_buff, car2_v_buff)
        car2_l_a.set_data(t_buff, car2_a_buff)
        car2_l_s.set_data(t_buff, car2_s_buff)

        for k in range(len(t_pred)-1):
            if t >= t_pred[k] and t <= t_pred[k+1]:
                car1_l_pred.set_data(car1_data[k+1]['pred'].x, car1_data[k+1]['pred'].y)
                car2_l_pred.set_data(car2_data[k+1]['pred'].x, car2_data[k+1]['pred'].y)
                # car1_l_ref.set_data(car1_data[k+1]['ref'][:,0], car1_data[k+1]['ref'][:,1])
                # car2_l_ref.set_data(car2_data[k+1]['ref'][:,0], car2_data[k+1]['ref'][:,1])
                break
        
        for k in range(len(t_game)-1):
            if t >= t_game[k] and t <= t_game[k+1]:
                car1_l_ref.set_data(game_data[k]['ref'][:,0], game_data[k]['ref'][:,1])
                car2_l_ref.set_data(game_data[k]['ref'][:,6], game_data[k]['ref'][:,7])
                break
        if t > t_game[-1]:
            car1_l_ref.set_data(game_data[-1]['ref'][:,0], game_data[-1]['ref'][:,1])
            car2_l_ref.set_data(game_data[-1]['ref'][:,6], game_data[-1]['ref'][:,7])

        ax_v.relim()
        ax_v.autoscale_view()
        ax_a.relim()
        ax_a.autoscale_view()
        ax_s.relim()
        ax_s.autoscale_view()

    writer.grab_frame()