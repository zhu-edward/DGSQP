#!/usr/bin/env python3

from DGSQP.dynamics.dynamics_simulator import DynamicsSimulator

from DGSQP.types import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from DGSQP.dynamics.model_types import DynamicBicycleConfig
from DGSQP.tracks.track_lib import get_track

from car1_tracking_controller_setup import mpc_controller as car1_tracking_controller
from car2_tracking_controller_setup import mpc_controller as car2_tracking_controller

from game_setup_unicycle import dgsqp_planner, pid_ws_fns

import pdb

import numpy as np
import casadi as ca

import copy
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import matplotlib
matplotlib.use('TkAgg')

def process_raceline(path, scaling):
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
    raceline = ca.Function('raceline', [t_sym], [ri(t_sym) for ri in raceline_interp])
    s2t = ca.interpolant('s2t', 'linear', [raceline_two_laps[:,6]], T_two_laps)

    return raceline, s2t, raceline_mat

plot = True
save = False
save_start = 0
save_steps = 200

save_trace = False
if save_trace:
    car1_trace = {'t': [],
                'x' : [],
                'y' : [],
                'psi' : [],
                's' : [],
                'e_y' : [],
                'e_psi' : [],
                'v_long' : [],
                'v_tran' : [],
                'psidot' : [],
                'u_a': [],
                'u_s': []}
    car2_trace = {'t': [],
                'x' : [],
                'y' : [],
                'psi' : [],
                's' : [],
                'e_y' : [],
                'e_psi' : [],
                'v_long' : [],
                'v_tran' : [],
                'psidot' : [],
                'u_a': [],
                'u_s': []}

# Initial time
t = 0

lookahead_time = 3
lookahead_window = 10

car1_obs_r = 0.21
car2_obs_r = 0.21

# Import scenario
track_name = 'L_track_barc'
track_obj = get_track(track_name)

L = track_obj.track_length
H = track_obj.half_width

car1_raceline_path = track_name + '_raceline_1.npz'
car1_raceline, car1_s2t, car1_raceline_mat = process_raceline(car1_raceline_path, 1.7)

car2_raceline_path = track_name + '_raceline_2.npz'
car2_raceline, car2_s2t, car2_raceline_mat = process_raceline(car2_raceline_path, 1.9)

# =============================================
# Set up model
# =============================================
discretization_method = 'rk4'
dt = 0.01
car1_sim_dynamics_config = DynamicBicycleConfig(dt=0.01,
                                        model_name='dynamic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=False,
                                        tire_model='pacejka',
                                        mass=2.2187,
                                        yaw_inertia=0.02723,
                                        wheel_friction=0.96,
                                        pacejka_b_front=0.99,
                                        pacejka_b_rear=0.99,
                                        pacejka_c_front=11.04,
                                        pacejka_c_rear=11.04)
car1_dynamics_simulator = DynamicsSimulator(t, car1_sim_dynamics_config, delay=None, track=track_obj)

car2_sim_dynamics_config = DynamicBicycleConfig(dt=0.01,
                                        model_name='dynamic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=False,
                                        tire_model='pacejka',
                                        mass=2.2187,
                                        yaw_inertia=0.02723,
                                        wheel_friction=0.96,
                                        pacejka_b_front=0.99,
                                        pacejka_b_rear=0.99,
                                        pacejka_c_front=11.04,
                                        pacejka_c_rear=11.04)
car2_dynamics_simulator = DynamicsSimulator(t, car2_sim_dynamics_config, delay=None, track=track_obj)

rng = np.random.default_rng()

# Green defends position
# car2_s = 11.811900809891037
# car2_raceline_init = np.array(car2_raceline(car2_s2t(car2_s))).squeeze()
# car2_ey = car2_raceline_init[7]
# car2_ep = car2_raceline_init[5]
# car2_vl = car2_raceline_init[2]
# car2_vt = car2_raceline_init[3]
# car2_w = car2_raceline_init[5]

# car1_s = 8.768197183420575
# car1_raceline_init = np.array(car1_raceline(car1_s2t(car1_s))).squeeze()
# car1_ey = car1_raceline_init[7]
# car1_ep = car1_raceline_init[5]
# car1_vl = car1_raceline_init[2]
# car1_vt = car1_raceline_init[3]
# car1_w = car1_raceline_init[5]

# car1_sim_state = VehicleState(t=0.0, 
#                                 p=ParametricPose(s=car1_s, x_tran=car1_ey, e_psi=car1_ep), 
#                                 v=BodyLinearVelocity(v_long=car1_vl, v_tran=car1_vt),
#                                 w=BodyAngularVelocity(w_psi=car1_w))
# car2_sim_state = VehicleState(t=0.0, 
#                                 p=ParametricPose(s=car2_s, x_tran=car2_ey, e_psi=car2_ep), 
#                                 v=BodyLinearVelocity(v_long=car2_vl, v_tran=car2_vt),
#                                 w=BodyAngularVelocity(w_psi=car2_w))

# Blue overtakes
car1_sim_state = VehicleState(t=0.0, 
                                p=ParametricPose(s=L-2.0, x_tran=0.1, e_psi=0), 
                                v=BodyLinearVelocity(v_long=0.5))
car2_sim_state = VehicleState(t=0.0, 
                                p=ParametricPose(s=2.0, x_tran=-0.1, e_psi=0), 
                                v=BodyLinearVelocity(v_long=0.5))

track_obj.local_to_global_typed(car1_sim_state)
track_obj.local_to_global_typed(car2_sim_state)

car1_init_state = copy.deepcopy(car1_sim_state)
car2_init_state = copy.deepcopy(car2_sim_state)

if plot:
    VL = 0.37
    VW = 0.195
    
    plt.ion()
    fig = plt.figure(figsize=(25,10))
    ax_xy = fig.add_subplot(1,2,1)
    ax_a = fig.add_subplot(2,2,2)
    ax_d = fig.add_subplot(2,2,4)
    track_obj.plot_map(ax_xy)
    ax_xy.plot(car1_raceline_mat[:,0], car1_raceline_mat[:,1], 'b')
    ax_xy.plot(car2_raceline_mat[:,0], car2_raceline_mat[:,1], 'g')
    # ax_xy.plot(x_ref, y_ref, 'r')
    ax_xy.set_aspect('equal')

    car1_l_pred = ax_xy.plot([], [], 'b-o', markersize=4)[0]
    car1_l_ref = ax_xy.plot([], [], 'bs', markersize=4, markerfacecolor='None')[0]
    car1_l_la = ax_xy.plot([], [], 'bs', markersize=4, markerfacecolor='None')[0]
    car1_l_a = ax_a.plot([], [], '-bo')[0]
    car1_l_d = ax_d.plot([], [], '-bo')[0]

    car2_l_pred = ax_xy.plot([], [], 'g-o', markersize=4)[0]
    car2_l_ref = ax_xy.plot([], [], 'gs', markersize=4, markerfacecolor='None')[0]
    car2_l_la = ax_xy.plot([], [], 'gs', markersize=4, markerfacecolor='None')[0]
    car2_l_a = ax_a.plot([], [], '-go')[0]
    car2_l_d = ax_d.plot([], [], '-go')[0]

    car1_rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='b', alpha=0.5)
    car2_rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='g', alpha=0.5)
    ax_xy.add_patch(car1_rect)
    ax_xy.add_patch(car2_rect)

    b_left = car1_sim_state.x.x - VL/2
    b_bot  = car1_sim_state.x.y - VW/2
    r = Affine2D().rotate_around(car1_sim_state.x.x, car1_sim_state.x.y, car1_sim_state.e.psi) + ax_xy.transData
    car1_rect.set_xy((b_left,b_bot))
    car1_rect.set_transform(r)

    b_left = car2_sim_state.x.x - VL/2
    b_bot  = car2_sim_state.x.y - VW/2
    r = Affine2D().rotate_around(car2_sim_state.x.x, car2_sim_state.x.y, car2_sim_state.e.psi) + ax_xy.transData
    car2_rect.set_xy((b_left,b_bot))
    car2_rect.set_transform(r)

    fig.canvas.draw()
    fig.canvas.flush_events()

pdb.set_trace()

control_dt = 0.1
game_dt = 0.1

# obs_r = 0.21
obs_p = np.array([-100, -100])

car1_N = car1_tracking_controller.N
car2_N = car2_tracking_controller.N
game_N = dgsqp_planner.N

car1_t = car1_s2t(car1_sim_state.p.s)
car1_t_ref = car1_t + control_dt*np.arange(car1_N+1)
car1_q_ref = np.array(car1_raceline(car1_t_ref)).squeeze().T

car1_u_ws = 0.01*np.ones((car1_N+1, car1_tracking_controller.dynamics.n_u))
car1_du_ws = np.zeros((car1_N, car1_tracking_controller.dynamics.n_u))

P = np.append(car1_q_ref.ravel(), np.tile(obs_p, car1_N))
car1_tracking_controller.set_warm_start(car1_u_ws, car1_du_ws, state=car1_sim_state, params=P)

car2_t = car2_s2t(car2_sim_state.p.s)
car2_t_ref = car2_t + control_dt*np.arange(car2_N+1)
car2_q_ref = np.array(car2_raceline(car2_t_ref)).squeeze().T

car2_u_ws = 0.01*np.ones((car2_N+1, car2_tracking_controller.dynamics.n_u))
car2_du_ws = np.zeros((car2_N, car2_tracking_controller.dynamics.n_u))

P = np.append(car2_q_ref.ravel(), np.tile(obs_p, car2_N))
car2_tracking_controller.set_warm_start(car2_u_ws, car2_du_ws, state=car2_sim_state, params=P)

if plot:
    car1_l_ref.set_data(car1_q_ref[:,0], car1_q_ref[:,1])
    car2_l_ref.set_data(car2_q_ref[:,0], car2_q_ref[:,1])

# =============================================
# Run race
# =============================================
# Initialize inputs
t = 0.0
car1_sim_state.u.u_a, car1_sim_state.u.u_steer = 0.0, 0.0
car2_sim_state.u.u_a, car2_sim_state.u.u_steer = 0.0, 0.0
car1_pred = VehiclePrediction()
car2_pred = VehiclePrediction()
car1_control = VehicleActuation(t=t, u_a=0, u_steer=0)
car2_control = VehicleActuation(t=t, u_a=0, u_steer=0)

solve_game = True

control_steps = 0
counter = 0

car1_lap_no = 0
car2_lap_no = 0
car1_use_game_sol = False
car2_use_game_sol = False

car1_data = []
car2_data = []
game_data = []

idx = 0
while True:
# for _ in range(100):
    print('=============================================')
    print(f't: {t}')
    car1_state = copy.deepcopy(car1_sim_state)
    car2_state = copy.deepcopy(car2_sim_state)

    if save_trace:
        car1_trace['t'].append(t)
        car1_trace['x'].append(car1_state.x.x)
        car1_trace['y'].append(car1_state.x.y)
        car1_trace['psi'].append(car1_state.e.psi)
        car1_trace['s'].append(car1_state.p.s)
        car1_trace['e_y'].append(car1_state.p.x_tran)
        car1_trace['e_psi'].append(car1_state.p.e_psi)
        car1_trace['v_long'].append(car1_state.v.v_long)
        car1_trace['v_tran'].append(car1_state.v.v_tran)
        car1_trace['psidot'].append(car1_state.w.w_psi)
        car1_trace['u_a'].append(car1_state.u.u_a)
        car1_trace['u_s'].append(car1_state.u.u_steer)

        car2_trace['t'].append(t)
        car2_trace['x'].append(car2_state.x.x)
        car2_trace['y'].append(car2_state.x.y)
        car2_trace['psi'].append(car2_state.e.psi)
        car2_trace['s'].append(car2_state.p.s)
        car2_trace['e_y'].append(car2_state.p.x_tran)
        car2_trace['e_psi'].append(car2_state.p.e_psi)
        car2_trace['v_long'].append(car2_state.v.v_long)
        car2_trace['v_tran'].append(car2_state.v.v_tran)
        car2_trace['psidot'].append(car2_state.w.w_psi)
        car2_trace['u_a'].append(car2_state.u.u_a)
        car2_trace['u_s'].append(car2_state.u.u_steer)

    # if car1_lap_no > car2_lap_no:
    #     car1_state.p.s += L
    # elif car2_lap_no > car1_lap_no:
    #     car2_state.p.s += L
    # print(car1_state.p.s, car2_state.p.s)

    # car1_use_game_sol = False # Uncomment to have car 1 only follow race line
    
    # Adjust car 1 reference if game solution is used
    st = time.time()
    if car1_use_game_sol:
        car1_q_ref = np.zeros((car1_N+1, 8))
        # Reference was computed before lap transition
        if car1_game_s_lim[0] - car1_state.p.s > L/2:
            car1_state.p.s += L
        if car1_state.p.s < car1_game_s_lim[0]:
            # Vehicle on raceline before game solution
            car1_t = float(car1_s2t(car1_state.p.s))
            car1_t_ref = car1_t + control_dt*np.arange(car1_N+1)
            for k in range(len(car1_t_ref)):
                if car1_t_ref[k] < car1_t_ref_game[0]:
                    # Case where time stamp corresponds to before game sol on raceline
                    car1_q_ref[k] = np.array(car1_raceline(car1_t_ref[k])).squeeze()
                elif car1_t_ref[k] >= car1_t_ref_game[0] and car1_t_ref[k] <= car1_t_ref_game[-1]:
                    # Case where time stamp is in the range of the game sol
                    car1_q_ref[k] = np.array(car1_game_line(car1_t_ref[k])).squeeze()
                elif car1_t_ref[k] > car1_t_ref_game[-1]:
                    # Case where time stamp corresponds to after game sol on raceline
                    t_bar = car1_t_ref[k] - car1_t_ref_game[-1] + car1_raceline_ts
                    car1_q_ref[k] = np.array(car1_raceline(t_bar)).squeeze()
                else:
                    pdb.set_trace()
        elif car1_state.p.s >= car1_game_s_lim[0] and car1_state.p.s <= car1_game_s_lim[1]:
            # Vehicle on game solution
            car1_t = float(car1_game_s2t(car1_state.p.s))
            car1_t_ref = car1_t + control_dt*np.arange(car1_N+1)
            for k in range(len(car1_t_ref)):
                if car1_t_ref[k] >= car1_t_ref_game[0] and car1_t_ref[k] <= car1_t_ref_game[-1]:
                    # Case where time stamp is in the range of the game sol
                    car1_q_ref[k] = np.array(car1_game_line(car1_t_ref[k])).squeeze()
                elif car1_t_ref[k] > car1_t_ref_game[-1]:
                    # Case where time stamp corresponds to after game sol on raceline
                    t_bar = car1_t_ref[k] - car1_t_ref_game[-1] + car1_raceline_ts
                    car1_q_ref[k] = np.array(car1_raceline(t_bar)).squeeze()
                else:
                    pdb.set_trace()
        else:
            # Vehicle on raceline after game solution
            car1_use_game_sol = False
    if not car1_use_game_sol:
        car1_t = float(car1_s2t(car1_state.p.s))
        car1_t_ref = car1_t + control_dt*np.arange(car1_N+1)
        car1_q_ref = np.array(car1_raceline(car1_t_ref)).squeeze().T
    car1_ref_time = time.time() - st

    # car2_use_game_sol = False # Uncomment to have car 2 only follow race line
    
    # Adjust car 2 reference if game solution is used
    st = time.time()
    if car2_use_game_sol:
        car2_q_ref = np.zeros((car2_N+1, 8))
        # Reference was computed before lap transition
        if car2_game_s_lim[0] - car2_state.p.s > L/2:
            car2_state.p.s += L
        if car2_state.p.s < car2_game_s_lim[0]:
            # Vehicle on raceline before game solution
            car2_t = float(car2_s2t(car2_state.p.s))
            car2_t_ref = car2_t + control_dt*np.arange(car2_N+1)
            for k in range(len(car2_t_ref)):
                if car2_t_ref[k] < car2_t_ref_game[0]:
                    # Case where time stamp corresponds to before game sol on raceline
                    car2_q_ref[k] = np.array(car2_raceline(car2_t_ref[k])).squeeze()
                elif car2_t_ref[k] >= car2_t_ref_game[0] and car2_t_ref[k] <= car2_t_ref_game[-1]:
                    # Case where time stamp is in the range of the game sol
                    car2_q_ref[k] = np.array(car2_game_line(car2_t_ref[k])).squeeze()
                elif car2_t_ref[k] > car2_t_ref_game[-1]:
                    # Case where time stamp corresponds to after game sol on raceline
                    t_bar = car2_t_ref[k] - car2_t_ref_game[-1] + car2_raceline_ts
                    car2_q_ref[k] = np.array(car2_raceline(t_bar)).squeeze()
                else:
                    pdb.set_trace()
        elif car2_state.p.s >= car2_game_s_lim[0] and car2_state.p.s <= car2_game_s_lim[1]:
            # Vehicle on game solution
            car2_t = float(car2_game_s2t(car2_state.p.s))
            car2_t_ref = car2_t + control_dt*np.arange(car2_N+1)
            for k in range(len(car2_t_ref)):
                if car2_t_ref[k] >= car2_t_ref_game[0] and car2_t_ref[k] <= car2_t_ref_game[-1]:
                    # Case where time stamp is in the range of the game sol
                    car2_q_ref[k] = np.array(car2_game_line(car2_t_ref[k])).squeeze()
                elif car2_t_ref[k] > car2_t_ref_game[-1]:
                    # Case where time stamp corresponds to after game sol on raceline
                    t_bar = car2_t_ref[k] - car2_t_ref_game[-1] + car2_raceline_ts
                    car2_q_ref[k] = np.array(car2_raceline(t_bar)).squeeze()
                else:
                    pdb.set_trace()
        else:
            # Vehicle on raceline after game solution
            car2_use_game_sol = False
    if not car2_use_game_sol:
        car2_t = float(car2_s2t(car2_state.p.s))
        car2_t_ref = car2_t + control_dt*np.arange(car2_N+1)
        car2_q_ref = np.array(car2_raceline(car2_t_ref)).squeeze().T
    car2_ref_time = time.time() - st
    
    if plot:
        car1_l_ref.set_data(car1_q_ref[:,0], car1_q_ref[:,1])
        car2_l_ref.set_data(car2_q_ref[:,0], car2_q_ref[:,1])
    
    # if car1_use_game_sol or car2_use_game_sol:
    #     pdb.set_trace()
    
    counter += 1
    if not car1_use_game_sol and not car2_use_game_sol and counter >= int(lookahead_time/control_dt):
        solve_game = True

    # Solve for car 1 control
    st = time.time()
    car2_p = car2_tracking_controller.q_pred[1:,:2].ravel()
    P = np.append(car1_q_ref.ravel(), car2_p)
    car1_tracking_controller.step(car1_state, params=P)
    car1_pred = car1_tracking_controller.get_prediction()
    car1_pred.t = t
    car1_solve_time = time.time()-st + car1_ref_time
    print('Car 1 controller solve time: ' + str(car1_solve_time))
    
    # Solve for car 2 control
    st = time.time()
    car1_p = car1_tracking_controller.q_pred[1:,:2].ravel()
    P = np.append(car2_q_ref.ravel(), car1_p)
    car2_tracking_controller.step(car2_state, params=P)
    car2_pred = car2_tracking_controller.get_prediction()
    car2_pred.t = t
    car2_solve_time = time.time()-st + car2_ref_time
    print('Car 2 controller solve time: ' + str(car2_solve_time))

    if save and idx >= save_start:
        car1_data.append(dict(t=t, state=copy.deepcopy(car1_state), pred=copy.deepcopy(car1_pred), ref=copy.deepcopy(car1_q_ref), solve_time=car1_solve_time))
        car2_data.append(dict(t=t, state=copy.deepcopy(car2_state), pred=copy.deepcopy(car2_pred), ref=copy.deepcopy(car2_q_ref), solve_time=car2_solve_time))

    if plot:
        b_left = car1_sim_state.x.x - VL/2
        b_bot  = car1_sim_state.x.y - VW/2
        r = Affine2D().rotate_around(car1_sim_state.x.x, car1_sim_state.x.y, car1_sim_state.e.psi) + ax_xy.transData
        car1_rect.set_xy((b_left,b_bot))
        car1_rect.set_transform(r)

        b_left = car2_sim_state.x.x - VL/2
        b_bot  = car2_sim_state.x.y - VW/2
        r = Affine2D().rotate_around(car2_sim_state.x.x, car2_sim_state.x.y, car2_sim_state.e.psi) + ax_xy.transData
        car2_rect.set_xy((b_left,b_bot))
        car2_rect.set_transform(r)

        if car1_pred is not None and car2_pred is not None:
            car1_l_pred.set_data(car1_pred.x, car1_pred.y)
            car2_l_pred.set_data(car2_pred.x, car2_pred.y)

            car1_l_a.set_data(np.arange(len(car1_pred.u_a)), car1_pred.u_a)
            car1_l_d.set_data(np.arange(len(car1_pred.u_steer)), car1_pred.u_steer)

            car2_l_a.set_data(np.arange(len(car2_pred.u_a)), car2_pred.u_a)
            car2_l_d.set_data(np.arange(len(car2_pred.u_steer)), car2_pred.u_steer)

        ax_a.relim()
        ax_a.autoscale_view()
        ax_d.relim()
        ax_d.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    car1_state.p.s = np.mod(car1_state.p.s, L)
    car2_state.p.s = np.mod(car2_state.p.s, L)
    
    # Solve game if required
    car1_pos = np.array([car1_state.x.x, car1_state.x.y])
    car2_pos = np.array([car2_state.x.x, car2_state.x.y])
    if solve_game:
        if np.linalg.norm(car1_pos - car1_q_ref[0,:2]) <= 0.2 and np.linalg.norm(car2_pos - car2_q_ref[0,:2]) <= 0.2:
            car1_t = float(car1_s2t(car1_state.p.s))
            car1_lookahead_t_ref = car1_t + lookahead_time + control_dt*np.arange(lookahead_window)
            car1_lookahead = np.array(car1_raceline(car1_lookahead_t_ref)).squeeze().T

            car2_t = float(car2_s2t(car2_state.p.s))
            car2_lookahead_t_ref = car2_t + lookahead_time + control_dt*np.arange(lookahead_window)
            car2_lookahead = np.array(car2_raceline(car2_lookahead_t_ref)).squeeze().T
            if plot:
                car1_l_la.set_data(car1_lookahead[:,0], car1_lookahead[:,1])
                car2_l_la.set_data(car2_lookahead[:,0], car2_lookahead[:,1])

            close = np.linalg.norm(car1_lookahead[:,:2]-car2_lookahead[:,:2], axis=1) <= 2.0*(car1_obs_r+car2_obs_r)
            free = np.linalg.norm(car1_lookahead[:,:2]-car2_lookahead[:,:2], axis=1) >= (car1_obs_r+car2_obs_r)
            interaction_idxs = np.argwhere(np.logical_and(close, free))
            if len(interaction_idxs) > 0:
                counter = 0
                solve_game = False
                game_solve_start = time.time()
                for i in interaction_idxs[0]:
                    car1_st = VehicleState(t=0)
                    car1_tracking_controller.dynamics.q2state(car1_st, car1_lookahead[i])
                    car2_st = VehicleState(t=0)
                    car2_tracking_controller.dynamics.q2state(car2_st, car2_lookahead[i])

                    car1_q, car1_u_ws = pid_ws_fns[0](game_N, car1_st, car1_st.v.v_long, car1_st.p.x_tran)
                    car2_q, car2_u_ws = pid_ws_fns[1](game_N, car2_st, car2_st.v.v_long, car2_st.p.x_tran)
                    dgsqp_planner.set_warm_start(np.hstack([car1_u_ws, car2_u_ws]))

                    info = dgsqp_planner.solve([car1_st, car2_st])

                    if info['msg'] in ['conv_abs_tol', 'conv_rel_tol']:
                        # pdb.set_trace()
                        q_ref_game = dgsqp_planner.q_pred
                        car1_q_bar = q_ref_game[:,:6]
                        car1_q_ref_game = np.zeros((q_ref_game.shape[0], 8))
                        car1_q_ref_game[:,:3] = car1_q_bar[:,:3]
                        car1_q_ref_game[:,5:] = car1_q_bar[:,3:]

                        car2_q_bar = q_ref_game[:,6:]
                        car2_q_ref_game = np.zeros((q_ref_game.shape[0], 8))
                        car2_q_ref_game[:,:3] = car2_q_bar[:,:3]
                        car2_q_ref_game[:,5:] = car2_q_bar[:,3:]

                        car1_t_ref_game = car1_lookahead_t_ref[i] + game_dt*np.arange(car1_q_ref_game.shape[0])
                        car2_t_ref_game = car2_lookahead_t_ref[i] + game_dt*np.arange(car2_q_ref_game.shape[0])

                        car1_game_s_lim = [car1_q_ref_game[0,6], car1_q_ref_game[-1,6]]
                        car2_game_s_lim = [car2_q_ref_game[0,6], car2_q_ref_game[-1,6]]

                        car1_raceline_ts = car1_s2t(car1_game_s_lim[1])
                        car2_raceline_ts = car2_s2t(car2_game_s_lim[1])

                        # Create interpolators for game solution
                        car1_game_interp = []
                        for i in range(car1_q_ref_game.shape[1]):
                            car1_game_interp.append(ca.interpolant(f'x{i}', 'linear', [car1_t_ref_game], car1_q_ref_game[:,i]))
                        t_sym = ca.MX.sym('t', 1)
                        car1_game_line = ca.Function('car1_raceline', [t_sym], [gi(t_sym) for gi in car1_game_interp])
                        car1_game_s2t = ca.interpolant('car1_s2t', 'linear', [car1_q_ref_game[:,6]], car1_t_ref_game)

                        car2_game_interp = []
                        for i in range(car2_q_ref_game.shape[1]):
                            car2_game_interp.append(ca.interpolant(f'x{i}', 'linear', [car2_t_ref_game], car2_q_ref_game[:,i]))
                        t_sym = ca.MX.sym('t', 1)
                        car2_game_line = ca.Function('car2_raceline', [t_sym], [gi(t_sym) for gi in car2_game_interp])
                        car2_game_s2t = ca.interpolant('car2_s2t', 'linear', [car2_q_ref_game[:,6]], car2_t_ref_game)
                        
                        car1_use_game_sol = True
                        car2_use_game_sol = True
                        if plot:
                            car1_l_la.set_data(car1_q_ref_game[:,0], car1_q_ref_game[:,1])
                            car2_l_la.set_data(car2_q_ref_game[:,0], car2_q_ref_game[:,1])

                        game_solve_time = time.time()-game_solve_start
                        print('Game solve time: ' + str(game_solve_time))

                        if save and idx >= save_start:
                            game_data.append(dict(t=t, ref=q_ref_game, info=info, solve_time=game_solve_time))
                        break

    t += control_dt
    idx += 1

    if save and idx >= save_start + save_steps:
        break
    
    # Step simulation forward
    car1_sim_state.u = car1_state.u
    car1_dynamics_simulator.step(car1_sim_state, T=control_dt)
    if car1_sim_state.p.s > L:
        car1_lap_no += 1
        car1_sim_state.p.s = np.mod(car1_sim_state.p.s, L)
    
    car2_sim_state.u = car2_state.u
    car2_dynamics_simulator.step(car2_sim_state, T=control_dt)
    if car2_sim_state.p.s > L:
        car2_lap_no += 1
        car2_sim_state.p.s = np.mod(car2_sim_state.p.s, L)

    # pdb.set_trace()

if save:
    data = dict(car1=car1_data, car2=car2_data, game=game_data)

    import pathlib
    import pickle
    from datetime import datetime

    time_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    data_dir = pathlib.Path(pathlib.Path.home(), f'results/comp_{time_str}')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    filename = 'data.pkl'
    data_path = data_dir.joinpath(filename)
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

    if save_trace:
        filename = 'car1_trace.npz'
        data_path = data_dir.joinpath(filename)
        np.savez(data_path, **car1_trace)

        filename = 'car2_trace.npz'
        data_path = data_dir.joinpath(filename)
        np.savez(data_path, **car2_trace)