#!/usr/bin/env python3

# This script runs the ablation experiment in Section VI.B.2 in https://arxiv.org/pdf/2203.16478.pdf

from DGSQP.solvers.IBR import IBR
from DGSQP.solvers.DGSQP import DGSQP
from DGSQP.solvers.PID import PIDLaneFollower

from DGSQP.solvers.solver_types import IBRParams, DGSQPParams, PIDParams

from DGSQP.types import VehicleState, VehicleActuation, Position, ParametricPose, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from DGSQP.dynamics.dynamics_models import CasadiKinematicBicycleCombined, CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.dynamics.model_types import KinematicBicycleConfig, MultiAgentModelConfig
from DGSQP.tracks.track_lib import *

import numpy as np
import casadi as ca

from datetime import datetime
import pathlib
import pickle

import copy

save_data = False

# Initial time
t = 0

if save_data:
    time_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    data_dir = pathlib.Path(pathlib.Path.home(), f'results/dgsqp_mc_ablation_{time_str}')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

car1_init_state=VehicleState(t=0.0, 
                            p=ParametricPose(s=1.5, x_tran=0.4, e_psi=0), 
                            v=BodyLinearVelocity(v_long=1.5))
car2_init_state=VehicleState(t=0.0, 
                            p=ParametricPose(s=1.0, x_tran=0.4, e_psi=0), 
                            v=BodyLinearVelocity(v_long=1.7))

# =============================================
# Helper functions
# =============================================
def check_collision(car1_traj, car2_traj, obs_d):
    for k in range(car1_traj.shape[0]):
        d = np.linalg.norm(car1_traj[k,:2] - car2_traj[k,:2], ord=2)
        if d < obs_d:
            return True
    return False

# Saturation cost fuction
sym_signed_u = ca.SX.sym('u', 1)
saturation_cost = ca.Function('saturation_cost', [sym_signed_u], [ca.fmax(ca.DM.zeros(1), sym_signed_u)])

dt = 0.1
discretization_method='euler'
half_width = 1.0

car1_dynamics_config = KinematicBicycleConfig(dt=dt,
                                            model_name='kinematic_bicycle_cl',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            wheel_dist_front=0.13,
                                            wheel_dist_rear=0.13,
                                            drag_coefficient=0.1,
                                            slip_coefficient=0.1,
                                            code_gen=False)

car2_dynamics_config = KinematicBicycleConfig(dt=dt,
                                            model_name='kinematic_bicycle_cl',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            wheel_dist_front=0.13,
                                            wheel_dist_rear=0.13,
                                            drag_coefficient=0.1,
                                            slip_coefficient=0.1,
                                            code_gen=False)

joint_model_config = MultiAgentModelConfig(dt=dt,
                                            discretization_method=discretization_method,
                                            use_mx=True,
                                            code_gen=False,
                                            verbose=True,
                                            compute_hessians=True)

car1_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
car1_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
car1_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=np.pi))
car1_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-np.pi))

car2_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
car2_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
car2_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=np.pi))
car2_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-np.pi))

state_input_ub = [car1_state_input_max, car2_state_input_max]
state_input_lb = [car1_state_input_min, car2_state_input_min]

car1_cost_params = dict(input_weight=[1.0, 1.0],
                         input_rate_weight=[1.0, 1.0],
                         comp_weights=[10.0, 5.0],
                         blocking_weight=0,
                         obs_weight=0,
                         obs_r=0.3)
car2_cost_params = dict(input_weight=[1.0, 1.0],
                        input_rate_weight=[1.0, 1.0],
                        comp_weights=[5.0, 10.0],
                        blocking_weight=1,
                        obs_weight=5,
                        obs_r=0.3)

car1_r=0.2
car2_r=0.2

use_ws=True
ibr_ws=False

exp_N = [15, 20, 25]
# exp_theta = np.arange(15, 91, 15)
# exp_N = [25]
exp_theta = [90]
num_mc = 200
rng = np.random.default_rng(seed=1)

for theta in exp_theta:
    track_obj = CurveTrack(enter_straight_length=1,
                            curve_length=8,
                            curve_swept_angle=theta*np.pi/180,
                            exit_straight_length=5,
                            width=half_width*2,
                            slack=0.8,
                            ccw=True)
    for N in exp_N:
        # =============================================
        # Set up joint model
        # =============================================
        car1_dyn_model = CasadiKinematicBicycleCombined(t, car1_dynamics_config, track=track_obj)
        car2_dyn_model = CasadiKinematicBicycleCombined(t, car2_dynamics_config, track=track_obj)
        joint_model = CasadiDecoupledMultiAgentDynamicsModel(t, [car1_dyn_model, car2_dyn_model], joint_model_config)

        # =============================================
        # Solver setup
        # =============================================
        sqgames_all_params = DGSQPParams(solver_name='sqgames_all',
                                            dt=dt,
                                            N=N,
                                            reg=1e-3,
                                            nonmono_ls=True,
                                            merit_function='stat_l1',
                                            line_search_iters=50,
                                            sqp_iters=50,
                                            p_tol=1e-3,
                                            d_tol=1e-3,
                                            beta=0.01,
                                            tau=0.5,
                                            verbose=False,
                                            debug_plot=False,
                                            pause_on_plot=True)

        # Vanilla
        sqgames_none_params = DGSQPParams(solver_name='sqgames_none',
                                            dt=dt,
                                            N=N,
                                            reg=1e-3,
                                            nonmono_ls=False,
                                            merit_function='stat',
                                            line_search_iters=50,
                                            sqp_iters=50,
                                            p_tol=1e-3,
                                            d_tol=1e-3,
                                            beta=0.01,
                                            tau=0.5,
                                            verbose=False,
                                            debug_plot=False,
                                            pause_on_plot=True)

        # Symbolic placeholder variables
        sym_q = ca.MX.sym('q', joint_model.n_q)
        sym_u_car1 = ca.MX.sym('u_car1', car1_dyn_model.n_u)
        sym_u_car2 = ca.MX.sym('u_car2', car2_dyn_model.n_u)
        sym_um_car1 = ca.MX.sym('um_car1', car1_dyn_model.n_u)
        sym_um_car2 = ca.MX.sym('um_car2', car2_dyn_model.n_u)

        car1_x_idx = 0
        car1_y_idx = 1
        car1_s_idx = 4
        car1_ey_idx = 5

        car2_x_idx = 6
        car2_y_idx = 7
        car2_s_idx = 10
        car2_ey_idx = 11

        ua_idx = 0
        us_idx = 1

        car1_pos = sym_q[[car1_x_idx, car1_y_idx]]
        car2_pos = sym_q[[car2_x_idx, car2_y_idx]]

        obs_cost_d = car1_cost_params['obs_r'] + car2_cost_params['obs_r']

        # Build symbolic cost functions
        car1_quad_input_cost = (1/2)*(car1_cost_params['input_weight'][0]*sym_u_car1[ua_idx]**2 \
                                + car1_cost_params['input_weight'][1]*sym_u_car1[us_idx]**2)
        car1_quad_input_rate_cost = (1/2)*(car1_cost_params['input_rate_weight'][0]*(sym_u_car1[ua_idx]-sym_um_car1[ua_idx])**2 \
                                    + car1_cost_params['input_rate_weight'][1]*(sym_u_car1[us_idx]-sym_um_car1[us_idx])**2)
        car1_blocking_cost = (1/2)*car1_cost_params['blocking_weight']*(sym_q[car1_ey_idx] - sym_q[car2_ey_idx])**2
        car1_obs_cost = (1/2)*car1_cost_params['obs_weight']*saturation_cost(obs_cost_d-ca.norm_2(car1_pos - car2_pos))**2

        car1_prog_cost = -car1_cost_params['comp_weights'][0]*sym_q[car1_s_idx]
        # car1_comp_cost = car1_cost_params['comp_weights'][1]*(sym_q[car2_s_idx]-sym_q[car1_s_idx])
        car1_comp_cost = car1_cost_params['comp_weights'][1]*ca.atan(sym_q[car2_s_idx]-sym_q[car1_s_idx])

        car1_sym_stage = car1_quad_input_cost \
                        + car1_quad_input_rate_cost \
                        + car1_blocking_cost \
                        + car1_obs_cost

        car1_sym_term = car1_prog_cost \
                        + car1_comp_cost \
                        + car1_blocking_cost \
                        + car1_obs_cost

        car1_sym_costs = []
        for k in range(N):
            car1_sym_costs.append(ca.Function(f'car1_stage_{k}', [sym_q, sym_u_car1, sym_um_car1], [car1_sym_stage],
                                        [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'car1_stage_cost_{k}']))
        car1_sym_costs.append(ca.Function('car1_term', [sym_q], [car1_sym_term],
                                    [f'q_{N}'], ['car1_term_cost']))

        car2_quad_input_cost = (1/2)*(car2_cost_params['input_weight'][0]*sym_u_car2[ua_idx]**2 \
                                + car2_cost_params['input_weight'][1]*sym_u_car2[us_idx]**2)
        car2_quad_input_rate_cost = (1/2)*(car2_cost_params['input_rate_weight'][0]*(sym_u_car2[ua_idx]-sym_um_car2[ua_idx])**2 \
                                    + car2_cost_params['input_rate_weight'][1]*(sym_u_car2[us_idx]-sym_um_car2[us_idx])**2)
        car2_blocking_cost = (1/2)*car2_cost_params['blocking_weight']*(sym_q[car1_ey_idx] - sym_q[car2_ey_idx])**2
        car2_obs_cost = (1/2)*car2_cost_params['obs_weight']*saturation_cost(obs_cost_d-ca.norm_2(car1_pos - car2_pos))**2

        car2_prog_cost = -car2_cost_params['comp_weights'][0]*sym_q[car2_s_idx]
        # car2_comp_cost = car2_cost_params['comp_weights'][1]*(sym_q[car1_s_idx]-sym_q[car2_s_idx])
        car2_comp_cost = car2_cost_params['comp_weights'][1]*ca.atan(sym_q[car1_s_idx]-sym_q[car2_s_idx])

        car2_sym_stage = car2_quad_input_cost \
                        + car2_quad_input_rate_cost \
                        + car2_blocking_cost \
                        + car2_obs_cost

        car2_sym_term = car2_prog_cost \
                        + car2_comp_cost \
                        + car2_blocking_cost \
                        + car2_obs_cost

        car2_sym_costs = []
        for k in range(N):
            car2_sym_costs.append(ca.Function(f'car2_stage_{k}', [sym_q, sym_u_car2, sym_um_car2], [car2_sym_stage],
                                        [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'car2_stage_cost_{k}']))
        car2_sym_costs.append(ca.Function('car2_term', [sym_q], [car2_sym_term],
                                    [f'q_{N}'], ['car2_term_cost']))

        sym_costs = [car1_sym_costs, car2_sym_costs]

        # Build symbolic constraints g_i(x, u, um) <= 0
        car1_input_rate_constr = ca.vertcat((sym_u_car1[ua_idx]-sym_um_car1[ua_idx]) - dt*car1_state_input_rate_max.u.u_a,
                                        dt*car1_state_input_rate_min.u.u_a - (sym_u_car1[ua_idx]-sym_um_car1[ua_idx]),
                                        (sym_u_car1[us_idx]-sym_um_car1[us_idx]) - dt*car1_state_input_rate_max.u.u_steer,
                                        dt*car1_state_input_rate_min.u.u_steer - (sym_u_car1[us_idx]-sym_um_car1[us_idx]))

        car2_input_rate_constr = ca.vertcat((sym_u_car2[ua_idx]-sym_um_car2[ua_idx]) - dt*car2_state_input_rate_max.u.u_a,
                                        dt*car2_state_input_rate_min.u.u_a - (sym_u_car2[ua_idx]-sym_um_car2[ua_idx]),
                                        (sym_u_car2[us_idx]-sym_um_car2[us_idx]) - dt*car2_state_input_rate_max.u.u_steer,
                                        dt*car2_state_input_rate_min.u.u_steer - (sym_u_car2[us_idx]-sym_um_car2[us_idx]))

        obs_d = car1_r + car2_r
        obs_avoid_constr = (obs_d)**2 - ca.bilin(ca.DM.eye(2), car1_pos - car2_pos,  car1_pos - car2_pos)

        car1_constr_stage = car1_input_rate_constr
        car1_constr_term = None

        car1_constrs = []
        for k in range(N):
            car1_constrs.append(ca.Function(f'car1_constrs_{k}', [sym_q, sym_u_car1, sym_um_car1], [car1_constr_stage]))
        if car1_constr_term is None:
            car1_constrs.append(None)
        else:
            car1_constrs.append(ca.Function(f'car1_constrs_{N}', [sym_q], [car1_constr_term]))

        car2_constr_stage = car2_input_rate_constr
        car2_constr_term = None

        # constr_stage = obs_avoid_constr
        constr_stage = ca.vertcat(car1_input_rate_constr, car2_input_rate_constr, obs_avoid_constr)
        constr_term = obs_avoid_constr

        car2_constrs = []
        for k in range(N):
            car2_constrs.append(ca.Function(f'car2_constrs_{k}', [sym_q, sym_u_car2, sym_um_car2], [car2_constr_stage]))
        if car2_constr_term is None:
            car2_constrs.append(None)
        else:
            car2_constrs.append(ca.Function(f'car2_constrs_{N}', [sym_q], [car2_constr_term]))

        shared_constr_stage = obs_avoid_constr
        shared_constr_term = obs_avoid_constr

        shared_constrs = []
        for k in range(N):
            if k == 0:
                shared_constrs.append(None)
            else:
                shared_constrs.append(ca.Function(f'shared_constrs_{k}', [sym_q, ca.vertcat(sym_u_car1, sym_u_car2), ca.vertcat(sym_um_car1, sym_um_car2)], [shared_constr_stage]))
        shared_constrs.append(ca.Function(f'shared_constrs_{N}', [sym_q], [shared_constr_term]))

        agent_constrs = [car1_constrs, car2_constrs]

        sqgames_all_controller = DGSQP(joint_model,
                                        sym_costs,
                                        agent_constrs,
                                        shared_constrs,
                                        {'ub': state_input_ub, 'lb': state_input_lb},
                                        sqgames_all_params)

        sqgames_none_controller = DGSQP(joint_model,
                                        sym_costs,
                                        agent_constrs,
                                        shared_constrs,
                                        {'ub': state_input_ub, 'lb': state_input_lb},
                                        sqgames_none_params)

        if ibr_ws:
            ibr_params = IBRParams(solver_name='ibr',
                                    dt=dt,
                                    N=N,
                                    line_search_iters=50,
                                    ibr_iters=1,
                                    use_ps=False,
                                    p_tol=1e-3,
                                    d_tol=1e-3,
                                    verbose=False,
                                    debug_plot=False,
                                    pause_on_plot=True)
            ibr_controller = IBR(joint_model,
                                sym_costs,
                                agent_constrs,
                                shared_constrs,
                                {'ub': state_input_ub, 'lb': state_input_lb},
                                ibr_params)

        first_seg_len = track_obj.cl_segs[0,0]

        sq_all_res = []
        sq_none_res = []

        for i in range(num_mc):
            print('========================================================')
            print(f'Curved track with {theta} degree turn, control horizon: {N}, trial: {i+1}')
            while True:
                car1_sim_state = copy.copy(car1_init_state)
                car2_sim_state = copy.copy(car2_init_state)

                car1_sim_state.p.s = max(0.1, rng.random()*first_seg_len)
                car1_sim_state.p.x_tran = rng.random()*half_width*2 - half_width
                car1_sim_state.v.v_long = rng.random()+2

                d = 2*np.pi*rng.random()
                car2_sim_state.p.s = car1_sim_state.p.s + 1.2*obs_d*np.cos(d)
                if car2_sim_state.p.s < 0 or car2_sim_state.p.s < car1_sim_state.p.s:
                    continue
                car2_sim_state.p.x_tran = car1_sim_state.p.x_tran + 1.2*obs_d*np.sin(d)
                if np.abs(car2_sim_state.p.x_tran) > half_width:
                    continue
                # car2_sim_state.p.s = rng.random()*first_seg_len/2
                # car2_sim_state.p.x_tran = rng.random()*half_width*2 - half_width
                car2_sim_state.v.v_long = rng.random()+2

                track_obj.local_to_global_typed(car1_sim_state)
                track_obj.local_to_global_typed(car2_sim_state)

                # =============================================
                # Warm start controller setup
                # =============================================
                if use_ws:
                    # Set up PID controllers for warm start
                    car1_steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                                                x_ref=car1_sim_state.p.x_tran,
                                                u_max=car1_state_input_max.u.u_steer,
                                                u_min=car1_state_input_min.u.u_steer,
                                                du_max=car1_state_input_rate_max.u.u_steer,
                                                du_min=car1_state_input_rate_min.u.u_steer)
                    car1_speed_params = PIDParams(dt=dt, Kp=1.0,
                                                x_ref=car1_sim_state.v.v_long,
                                                u_max=car1_state_input_max.u.u_a,
                                                u_min=car1_state_input_min.u.u_a,
                                                du_max=car1_state_input_rate_max.u.u_a,
                                                du_min=car1_state_input_rate_min.u.u_a)
                    car1_pid_controller = PIDLaneFollower(dt, car1_steer_params, car1_speed_params)

                    car2_steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                                                x_ref=car2_sim_state.p.x_tran,
                                                u_max=car2_state_input_max.u.u_steer,
                                                u_min=car2_state_input_min.u.u_steer,
                                                du_max=car2_state_input_rate_max.u.u_steer,
                                                du_min=car2_state_input_rate_min.u.u_steer)
                    car2_speed_params = PIDParams(dt=dt, Kp=1.0,
                                                x_ref=car2_sim_state.v.v_long,
                                                u_max=car2_state_input_max.u.u_a,
                                                u_min=car2_state_input_min.u.u_a,
                                                du_max=car2_state_input_rate_max.u.u_a,
                                                du_min=car2_state_input_rate_min.u.u_a)
                    car2_pid_controller = PIDLaneFollower(dt, car2_steer_params, car2_speed_params)

                    # Construct initial guess for ALGAMES MPC with PID
                    car1_state = [copy.deepcopy(car1_sim_state)]
                    for i in range(N):
                        state = copy.deepcopy(car1_state[-1])
                        car1_pid_controller.step(state)
                        car1_dyn_model.step(state)
                        car1_state.append(state)

                    car2_state = [copy.deepcopy(car2_sim_state)]
                    for i in range(N):
                        state = copy.deepcopy(car2_state[-1])
                        car2_pid_controller.step(state)
                        car2_dyn_model.step(state)
                        car2_state.append(state)

                    car1_q_ws = np.zeros((N+1, car1_dyn_model.n_q))
                    car2_q_ws = np.zeros((N+1, car2_dyn_model.n_q))
                    car1_u_ws = np.zeros((N, car1_dyn_model.n_u))
                    car2_u_ws = np.zeros((N, car2_dyn_model.n_u))
                    for i in range(N+1):
                        car1_q_ws[i] = np.array([car1_state[i].x.x, car1_state[i].x.y, car1_state[i].v.v_long, car1_state[i].p.e_psi, car1_state[i].p.s-1e-6, car1_state[i].p.x_tran])
                        car2_q_ws[i] = np.array([car2_state[i].x.x, car2_state[i].x.y, car2_state[i].v.v_long, car2_state[i].p.e_psi, car2_state[i].p.s-1e-6, car2_state[i].p.x_tran])
                        if i < N:
                            car1_u_ws[i] = np.array([car1_state[i+1].u.u_a, car1_state[i+1].u.u_steer])
                            car2_u_ws[i] = np.array([car2_state[i+1].u.u_a, car2_state[i+1].u.u_steer])

                    collision = check_collision(car1_q_ws, car2_q_ws, obs_d)
                    if not collision:
                        break

                if ibr_ws:
                    ibr_controller.set_warm_start([car1_u_ws, car2_u_ws])

            # =============================================
            # Run for a single step
            # =============================================
            # Initialize inputs
            car1_sim_state.u.u_a, car1_sim_state.u.u_steer = 0.0, 0.0
            car2_sim_state.u.u_a, car2_sim_state.u.u_steer = 0.0, 0.0
            joint_state = [car1_sim_state, car2_sim_state]

            if ibr_ws:
                ibr_controller.step(copy.deepcopy(joint_state))
                sqgames_all_controller.set_warm_start(ibr_controller.u_pred)
                sqgames_none_controller.set_warm_start(ibr_controller.u_pred)
            else:
                sqgames_all_controller.set_warm_start(np.hstack([car1_u_ws, car2_u_ws]))
                sqgames_none_controller.set_warm_start(np.hstack([car1_u_ws, car2_u_ws]))

            sqgames_all_info = sqgames_all_controller.solve(copy.deepcopy(joint_state))
            sqgames_all_res = {'solve_info': copy.deepcopy(sqgames_all_info),
                            'params': copy.deepcopy(sqgames_all_params),
                            'init': copy.deepcopy(joint_state)}

            sqgames_none_info = sqgames_none_controller.solve(copy.deepcopy(joint_state))
            sqgames_none_res = {'solve_info': copy.deepcopy(sqgames_none_info),
                            'params': copy.deepcopy(sqgames_none_params),
                            'init': copy.deepcopy(joint_state)}

            sq_all_res.append(sqgames_all_res)
            sq_none_res.append(sqgames_none_res)

        results = dict(sqgames_all=sq_all_res,
                        sqgames_none=sq_none_res,
                        track=track_obj,
                        agent_dyn_configs=[car1_dynamics_config, car2_dynamics_config],
                        joint_model_config=joint_model_config)

        if save_data:
            filename = f'data_c_{theta}_N_{N}.pkl'
            data_path = data_dir.joinpath(filename)
            with open(data_path, 'wb') as f:
                pickle.dump(results, f)
