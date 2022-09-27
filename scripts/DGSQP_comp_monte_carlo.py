#!/usr/bin/env python3

# This script runs the race track experiment in Section VI.B in https://arxiv.org/pdf/2203.16478.pdf

from DGSQP.solvers.IBR import IBR
from DGSQP.solvers.DGSQP import DGSQP
from DGSQP.solvers.ALGAMES import ALGAMES
from DGSQP.solvers.PID import PIDLaneFollower

from DGSQP.solvers.solver_types import IBRParams, DGSQPParams, ALGAMESParams, PIDParams

from DGSQP.types import VehicleState, VehicleActuation, Position, ParametricPose, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from DGSQP.dynamics.dynamics_models import CasadiKinematicBicycleCombined, CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.dynamics.model_types import KinematicBicycleConfig, MultiAgentModelConfig
from DGSQP.tracks.track_lib import get_track

import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import pdb
import copy
import pickle
from datetime import datetime
import pathlib

save_data = False

if save_data:
    time_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    data_dir = pathlib.Path(pathlib.Path.home(), f'results/dgsqp_algames_mc_comp_{time_str}')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

def check_collision(agent_trajs, agent_rs):
    M = len(agent_trajs)
    for i in range(M):
        for j in range(i+1, M):
            obs_d = agent_rs[i] + agent_rs[j]
            for k in range(agent_trajs[i].shape[0]):
                d = np.linalg.norm(agent_trajs[i][k,:2] - agent_trajs[j][k,:2], ord=2)
                if d < obs_d:
                    return True
    return False

def check_boundaries(agent_trajs, H):
    M = len(agent_trajs)
    for i in range(M):
        for k in range(agent_trajs[i].shape[0]):
            ey = agent_trajs[i][k,5]
            if np.abs(ey) > H:
                return True
    return False

# Initial time
t = 0

dt = 0.1
N = 15

use_ws = True
ibr_ws = True

track_obj = get_track('L_track_barc')
H = track_obj.half_width
L = track_obj.track_length

def plot_env(ax):
    track_obj.plot_map(ax, close_loop=True)

# =============================================
# Set up joint model
# =============================================
discretization_method='euler'
car1_dynamics_config = KinematicBicycleConfig(dt=dt,
                                            model_name='kinematic_bicycle',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            wheel_dist_front=0.13,
                                            wheel_dist_rear=0.13,
                                            code_gen=False,
                                            M=1)
car1_dyn_model = CasadiKinematicBicycleCombined(t, car1_dynamics_config, track=track_obj)

car2_dynamics_config = KinematicBicycleConfig(dt=dt,
                                            model_name='kinematic_bicycle',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            wheel_dist_front=0.13,
                                            wheel_dist_rear=0.13,
                                            code_gen=False,
                                            M=1)
car2_dyn_model = CasadiKinematicBicycleCombined(t, car2_dynamics_config, track=track_obj)

car1_joint_config = MultiAgentModelConfig(dt=dt,
                                    discretization_method=discretization_method,
                                    use_mx=False,
                                    code_gen=False,
                                    verbose=True,
                                    compute_hessians=True,
                                    M=1)
joint_model = CasadiDecoupledMultiAgentDynamicsModel(t, [car1_dyn_model, car2_dyn_model], car1_joint_config)

car1_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=(H-0.1), e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
car1_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-(H-0.1), e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
car1_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
car1_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))

car2_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=H-0.1, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
car2_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-(H-0.1), e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
car2_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
car2_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))

state_input_ub = [car1_state_input_max, car2_state_input_max]
state_input_lb = [car1_state_input_min, car2_state_input_min]

car1_cost_params = dict(input_weight=[0.1, 0.1],
                         input_rate_weight=[1.0, 1.0],
                         comp_weights=[0, 1.0])
car2_cost_params = dict(input_weight=[0.1, 0.1],
                        input_rate_weight=[1.0, 1.0],
                        comp_weights=[0, 1.0])

car1_r=0.2
car2_r=0.2

# =============================================
# Helper functions
# =============================================
# Saturation cost fuction
sym_signed_u = ca.SX.sym('u', 1)
saturation_cost = ca.Function('saturation_cost', [sym_signed_u], [ca.fmax(ca.DM.zeros(1), sym_signed_u)])

# =============================================
# SQGAMES controller setup
# =============================================
dgsqp_params = DGSQPParams(solver_name='DGSQP',
                            dt=dt,
                            N=N,
                            merit_function='stat_l1',
                            nonmono_ls=True,
                            line_search_iters=50,
                            sqp_iters=50,
                            p_tol=1e-3,
                            d_tol=1e-3,
                            reg=0.0,
                            beta=0.01,
                            tau=0.5,
                            verbose=False,
                            debug_plot=False,
                            pause_on_plot=True)
algames_params = ALGAMESParams(solver_name='ALGAMES',
                                dt=dt,
                                N=N,
                                dynamics_hessians=True,
                                outer_iters=50,
                                line_search_iters=50,
                                line_search_tol=1e-6,
                                newton_iters=50,
                                newton_step_tol=1e-9,
                                ineq_tol=1e-3,
                                eq_tol=1e-3,
                                opt_tol=1e-3,
                                rho=1.0,
                                gamma=1.5,
                                rho_max=1e7,
                                beta=0.01,
                                tau=0.5,
                                q_reg=0.0,
                                u_reg=0.0,
                                verbose=False,
                                debug_plot=False,
                                pause_on_plot=False)

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

# Build symbolic cost functions
car1_quad_input_cost = (1/2)*(car1_cost_params['input_weight'][0]*sym_u_car1[ua_idx]**2 \
                         + car1_cost_params['input_weight'][1]*sym_u_car1[us_idx]**2)
car1_quad_input_rate_cost = (1/2)*(car1_cost_params['input_rate_weight'][0]*(sym_u_car1[ua_idx]-sym_um_car1[ua_idx])**2 \
                              + car1_cost_params['input_rate_weight'][1]*(sym_u_car1[us_idx]-sym_um_car1[us_idx])**2)

car1_prog_cost = -car1_cost_params['comp_weights'][0]*sym_q[car1_s_idx]
# car1_comp_cost = car1_cost_params['comp_weights'][1]*(sym_q[car2_s_idx]-sym_q[car1_s_idx])
car1_comp_cost = car1_cost_params['comp_weights'][1]*ca.atan(sym_q[car2_s_idx]-sym_q[car1_s_idx])

car1_sym_stage = car1_quad_input_cost \
                + car1_quad_input_rate_cost

car1_sym_term = car1_prog_cost \
                + car1_comp_cost

car1_sym_costs = []
for k in range(N):
    car1_sym_costs.append(ca.Function(f'car1_stage_{k}', [sym_q, sym_u_car1, sym_um_car1], [car1_sym_stage]))
car1_sym_costs.append(ca.Function('car1_term', [sym_q], [car1_sym_term]))

car2_quad_input_cost = (1/2)*(car2_cost_params['input_weight'][0]*sym_u_car2[ua_idx]**2 \
                         + car2_cost_params['input_weight'][1]*sym_u_car2[us_idx]**2)
car2_quad_input_rate_cost = (1/2)*(car2_cost_params['input_rate_weight'][0]*(sym_u_car2[ua_idx]-sym_um_car2[ua_idx])**2 \
                              + car2_cost_params['input_rate_weight'][1]*(sym_u_car2[us_idx]-sym_um_car2[us_idx])**2)

car2_prog_cost = -car2_cost_params['comp_weights'][0]*sym_q[car2_s_idx]
# car2_comp_cost = car2_cost_params['comp_weights'][1]*(sym_q[car1_s_idx]-sym_q[car2_s_idx])
car2_comp_cost = car2_cost_params['comp_weights'][1]*ca.atan(sym_q[car1_s_idx]-sym_q[car2_s_idx])

car2_sym_stage = car2_quad_input_cost \
                + car2_quad_input_rate_cost

car2_sym_term = car2_prog_cost \
                + car2_comp_cost

car2_sym_costs = []
for k in range(N):
    car2_sym_costs.append(ca.Function(f'car2_stage_{k}', [sym_q, sym_u_car2, sym_um_car2], [car2_sym_stage]))
car2_sym_costs.append(ca.Function('car2_term', [sym_q], [car2_sym_term]))

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

joint_constr_stage_0 = ca.vertcat(car1_input_rate_constr, car2_input_rate_constr)
joint_constr_stage = ca.vertcat(car1_input_rate_constr, car2_input_rate_constr, obs_avoid_constr)

joint_constr_term = obs_avoid_constr

u = ca.vertcat(sym_u_car1, sym_u_car2)
um = ca.vertcat(sym_um_car1, sym_um_car2)
joint_constrs = []
for k in range(N):
    if k == 0:
        if joint_constr_stage_0 is None:
            joint_constrs.append(None)
        else:
            joint_constrs.append(ca.Function(f'nl_constrs_{k}', [sym_q, u, um], [joint_constr_stage_0]))
    else:
        if joint_constr_stage is None:
            joint_constrs.append(None)
        else:
            joint_constrs.append(ca.Function(f'nl_constrs_{k}', [sym_q, u, um], [joint_constr_stage]))
if joint_constr_term is None:
    joint_constrs.append(None)
else:
    joint_constrs.append(ca.Function(f'nl_constrs_{N}', [sym_q], [joint_constr_term]))

joint_costs_stage = car1_sym_stage + car2_sym_stage
joint_costs_term = car1_sym_term + car2_sym_term

joint_sym_costs = []
for k in range(N):
    joint_sym_costs.append(ca.Function(f'joint_stage_{k}', [sym_q, u, um], [joint_costs_stage]))
joint_sym_costs.append(ca.Function('joint_term', [sym_q], [joint_costs_term]))

dgsqp_controller = DGSQP(joint_model, 
                            sym_costs, 
                            agent_constrs,
                            shared_constrs,
                            {'ub': state_input_ub, 'lb': state_input_lb},
                            dgsqp_params,
                            xy_plot=plot_env)

algames_controller = ALGAMES(joint_model, 
                                sym_costs, 
                                joint_constrs, 
                                {'ub': state_input_ub, 'lb': state_input_lb},
                                algames_params,
                                xy_plot=plot_env)

rng = np.random.default_rng(seed=0)

samples = 1000
i = 0
while i < samples:
    car1_s = L*rng.random()
    car1_ey = (H-0.1)*(2*rng.random()-1)
    car1_vx = 2.0 + (rng.random()-0.5) # Longitudinal velocity between 1.5 and 2.5 m/s
    car1_ep = 5.0*(2*rng.random()-1)*np.pi/180 # Heading deviation between -5 and 5 degrees from centerline tangent
    car1_sim_state = VehicleState(t=0.0, 
                                    p=ParametricPose(s=car1_s, x_tran=car1_ey, e_psi=car1_ep), 
                                    v=BodyLinearVelocity(v_long=car1_vx))
    
    car2_s = car1_s + 1.2*obs_d*(2*rng.random()-1)
    car2_ey = (H-0.1)*(2*rng.random()-1)
    car2_vx = (1+0.25*(2*rng.random()-1))*car1_vx
    car2_ep = 5.0*(2*rng.random()-1)*np.pi/180
    car2_sim_state = VehicleState(t=0.0, 
                                    p=ParametricPose(s=car2_s, x_tran=car2_ey, e_psi=car2_ep), 
                                    v=BodyLinearVelocity(v_long=car2_vx))

    track_obj.local_to_global_typed(car1_sim_state)
    track_obj.local_to_global_typed(car2_sim_state)

    car1_init_state = copy.deepcopy(car1_sim_state)
    car2_init_state = copy.deepcopy(car2_sim_state)

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

        # Construct initial guess for with PID
        car1_state = [copy.deepcopy(car1_sim_state)]
        for _ in range(N):
            state = copy.deepcopy(car1_state[-1])
            car1_pid_controller.step(state)
            car1_dyn_model.step(state)
            car1_state.append(state)
            
        car2_state = [copy.deepcopy(car2_sim_state)]
        for _ in range(N):
            state = copy.deepcopy(car2_state[-1])
            car2_pid_controller.step(state)
            car2_dyn_model.step(state)
            car2_state.append(state)
        
        car1_q_ws = np.zeros((N+1, car1_dyn_model.n_q))
        car2_q_ws = np.zeros((N+1, car2_dyn_model.n_q))
        car1_u_ws = np.zeros((N, car1_dyn_model.n_u))
        car2_u_ws = np.zeros((N, car2_dyn_model.n_u))
        for k in range(N+1):
            car1_q_ws[k] = np.array([car1_state[k].x.x, car1_state[k].x.y, car1_state[k].v.v_long, car1_state[k].p.e_psi, car1_state[k].p.s-1e-6, car1_state[k].p.x_tran])
            car2_q_ws[k] = np.array([car2_state[k].x.x, car2_state[k].x.y, car2_state[k].v.v_long, car2_state[k].p.e_psi, car2_state[k].p.s-1e-6, car2_state[k].p.x_tran])
            if k < N:
                car1_u_ws[k] = np.array([car1_state[k+1].u.u_a, car1_state[k+1].u.u_steer])
                car2_u_ws[k] = np.array([car2_state[k+1].u.u_a, car2_state[k+1].u.u_steer])

        if check_collision([car1_q_ws, car2_q_ws], [car1_r, car2_r]):
            continue
        # if check_boundaries([car1_q_ws, car2_q_ws], H):
        #     continue

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
            ibr_controller.set_warm_start([car1_u_ws, car2_u_ws])
            ibr_controller.step([car1_sim_state, car2_sim_state])
            if np.all([s.stats()['success'] for s in ibr_controller.br_solvers]):
                dgsqp_controller.set_warm_start(ibr_controller.u_pred)
                algames_controller.set_warm_start(ibr_controller.q_pred, ibr_controller.u_pred)
            else:
                dgsqp_controller.set_warm_start(np.hstack([car1_u_ws, car2_u_ws]))
                algames_controller.set_warm_start(np.hstack([car1_q_ws, car2_q_ws]), np.hstack([car1_u_ws, car2_u_ws]))
        else:
            dgsqp_controller.set_warm_start(np.hstack([car1_u_ws, car2_u_ws]))
            algames_controller.set_warm_start(np.hstack([car1_q_ws, car2_q_ws]), np.hstack([car1_u_ws, car2_u_ws]))

    print(f'Sample {i+1}')

    # =============================================
    # Run for a single step
    # =============================================
    # Initialize inputs
    car1_sim_state.u.u_a, car1_sim_state.u.u_steer = 0.0, 0.0
    car2_sim_state.u.u_a, car2_sim_state.u.u_steer = 0.0, 0.0
    joint_state = [car1_sim_state, car2_sim_state]

    dgsqp_info = dgsqp_controller.solve(copy.deepcopy(joint_state))
    dgsqp_res = {'solve_info': copy.deepcopy(dgsqp_info), 
                    'params': copy.deepcopy(dgsqp_params), 
                    'init': copy.deepcopy(joint_state)}

    algames_info = algames_controller.solve(copy.deepcopy(joint_state))
    algames_res = {'solve_info': copy.deepcopy(algames_info), 
                    'params': copy.deepcopy(algames_params), 
                    'init': copy.deepcopy(joint_state)}

    if save_data:
        results = dict(dgsqp=dgsqp_res, algames=algames_res, env=track_obj)
        filename = f'sample_{i+1}.pkl'
        data_path = data_dir.joinpath(filename)
        with open(data_path, 'wb') as f:
            pickle.dump(results, f)

    i += 1