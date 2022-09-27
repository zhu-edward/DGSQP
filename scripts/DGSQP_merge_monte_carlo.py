#!/usr/bin/env python3

# This script runs the merge experiment in Section VI.A in https://arxiv.org/pdf/2203.16478.pdf

from DGSQP.solvers.DGSQP import DGSQP
from DGSQP.solvers.ALGAMES import ALGAMES
from DGSQP.solvers.solver_types import DGSQPParams, ALGAMESParams

from DGSQP.types import VehicleState, VehicleActuation, Position, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from DGSQP.dynamics.dynamics_models import CasadiKinematicUnicycle, CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.dynamics.model_types import DynamicsConfig, MultiAgentModelConfig

import numpy as np
import casadi as ca

from datetime import datetime
import pathlib
import copy
import pickle

save_data = False

if save_data:
    time_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    data_dir = pathlib.Path(pathlib.Path.home(), f'results/dgsqp_algames_mc_merge_{time_str}')
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

# Define lane geometry
ll = 5 # Straight lane length
lw = 0.3 # Straight lane width
mw = 0.3 # Ramp width
mp = 1.5 # x-coordinate where ramp intersects with straight lane
th = np.pi/12 # Angle of intersection
r = 0.1

ns = ca.DM([0, 1])
nm = ca.DM([-np.sin(th), np.cos(th)])

# Straight left boundary
x1 = ca.DM([0, lw])
x2 = ca.DM([ll, lw])
# Straight right boundary
x3 = ca.DM([0, 0])
x4 = ca.DM([ll, 0])
# Ramp left boundary
x5 = ca.DM([mp,  0])
x6 = ca.DM([mp+lw/np.tan(th),  lw])
# Ramp right boundary
x7 = ca.DM([mp+mw/np.sin(th), 0])
x8 = ca.DM([x7[0]+lw/np.tan(th), lw])

merge_env = dict(straight_left=[x1, x2], straight_right=[x3, x4], ramp_left=[x5, x6], ramp_right=[x7, x8])

p = ca.SX.sym('p', 2)
straight_lane_1 = ca.Function('straight_lane_1', [p], [ns.T @ (p-(x1-r*ns))])
straight_lane_2 = ca.Function('straight_lane_2', [p], [-ns.T @ (p-(x3+r*ns))])

ramp_lane_left_n = ca.Function('ramp_lane_left_n', [p[0]], [ca.vertcat(ca.pw_const(p[0], ca.DM([x6[0]]), ca.DM([nm[0], ns[0]])), ca.pw_const(p[0], ca.DM([x6[0]]), ca.DM([nm[1], ns[1]])))])
ramp_lane_right_n = ca.Function('ramp_lane_right_n1', [p[0]], [ca.vertcat(ca.pw_const(p[0], ca.DM([x7[0]]), -ca.DM([nm[0], ns[0]])), ca.pw_const(p[0], ca.DM([x7[0]]), -ca.DM([nm[1], ns[1]])))])

ramp_lane_1 = ca.Function('ramp_lane_1', [p], [ramp_lane_left_n(p[0]).T @ (p-(x6-r*ramp_lane_left_n(p[0])))])
ramp_lane_2 = ca.Function('ramp_lane_2', [p], [ramp_lane_right_n(p[0]).T @ (p-(x7-r*ramp_lane_right_n(p[0])))])

def plot_env(ax):
    ax.plot([float(x1[0]), float(x2[0])], [float(x1[1]), float(x2[1])], 'k', linewidth=1.5)
    ax.plot([float(x3[0]), float(x5[0])], [float(x3[1]), float(x5[1])], 'k', linewidth=1.5)
    ax.plot([float(x7[0]), float(x4[0])], [float(x7[1]), float(x4[1])], 'k', linewidth=1.5)

    ax.plot([float(x1[0]), float(x5[0])], [float(-x5[0])*np.tan(th), float(x5[1])], 'k', linewidth=1.5)
    ax.plot([float(x1[0]), float(x7[0])], [float(-x7[0])*np.tan(th), float(x7[1])], 'k', linewidth=1.5)

# Initial time
t = 0

dt = 0.1
N = 20

car1_goal = np.array([4.0, 0.15, 0.3, 0])
car2_goal = np.array([4.5, 0.15, 0.3, 0])
car3_goal = np.array([4.25, 0.15, 0.3, 0])

# Set up dynamics models
discretization_method='rk3'
car1_dynamics_config = DynamicsConfig(dt=dt,
                                        model_name='kinematic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        code_gen=False,
                                        M=1)
car1_dyn_model = CasadiKinematicUnicycle(t, car1_dynamics_config)

car2_dynamics_config = DynamicsConfig(dt=dt,
                                        model_name='kinematic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        code_gen=False,
                                        M=1)
car2_dyn_model = CasadiKinematicUnicycle(t, car2_dynamics_config)

car3_dynamics_config = DynamicsConfig(dt=dt,
                                        model_name='kinematic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        code_gen=False,
                                        M=1)
car3_dyn_model = CasadiKinematicUnicycle(t, car3_dynamics_config)

joint_config = MultiAgentModelConfig(dt=dt,
                                    discretization_method=discretization_method,
                                    use_mx=False,
                                    code_gen=False,
                                    verbose=True,
                                    compute_hessians=True,
                                    M=1)
joint_model = CasadiDecoupledMultiAgentDynamicsModel(t, [car1_dyn_model, car2_dyn_model, car3_dyn_model], joint_config)

# State and input bounds
car1_state_input_max = VehicleState(x=Position(x=np.inf, y=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=2.0, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.0, u_steer=4.5))
car1_state_input_min = VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-2.0, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.0, u_steer=-4.5))

car2_state_input_max = VehicleState(x=Position(x=np.inf, y=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=2.0, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.0, u_steer=4.5))
car2_state_input_min = VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-2.0, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.0, u_steer=-4.5))

car3_state_input_max = VehicleState(x=Position(x=np.inf, y=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=2.0, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.0, u_steer=4.5))
car3_state_input_min = VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-2.0, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.0, u_steer=-4.5))

state_input_ub = [car1_state_input_max, car2_state_input_max, car3_state_input_max]
state_input_lb = [car1_state_input_min, car2_state_input_min, car3_state_input_min]

# Collision buffer
car1_r=0.1
car2_r=0.1
car3_r=0.1

# =============================================
# Helper functions
# =============================================
# Saturation cost fuction
sym_signed_u = ca.SX.sym('u', 1)
saturation_cost = ca.Function('saturation_cost', [sym_signed_u], [ca.fmax(ca.DM.zeros(1), sym_signed_u)])

# Solver parameters
dgsqp_params = DGSQPParams(solver_name='DGSQP',
                            dt=dt,
                            N=N,
                            reg=0,
                            merit_function='stat_l1',
                            nonmono_ls=True,
                            line_search_iters=50,
                            sqp_iters=50,
                            p_tol=1e-3,
                            d_tol=1e-3,
                            beta=0.01,
                            tau=0.5,
                            verbose=False,
                            debug_plot=False,
                            pause_on_plot=True)
# algames_params = ALGAMESParams(solver_name='ALGAMES',
#                                 dt=dt,
#                                 N=N,
#                                 dynamics_hessians=False,
#                                 outer_iters=50,
#                                 line_search_iters=50,
#                                 line_search_tol=1e-6,
#                                 newton_iters=50,
#                                 newton_step_tol=1e-9,
#                                 ineq_tol=1e-3,
#                                 eq_tol=1e-3,
#                                 opt_tol=1e-3,
#                                 rho=1.0,
#                                 gamma=5.0,
#                                 rho_max=1e7,
#                                 beta=0.01,
#                                 tau=0.5,
#                                 q_reg=0.0,
#                                 u_reg=0.0,
#                                 verbose=False,
#                                 debug_plot=False,
#                                 pause_on_plot=False)

# Symbolic placeholder variables
sym_q = ca.MX.sym('q', joint_model.n_q)
sym_u_car1 = ca.MX.sym('u_car1', car1_dyn_model.n_u)
sym_u_car2 = ca.MX.sym('u_car2', car2_dyn_model.n_u)
sym_u_car3 = ca.MX.sym('u_car3', car3_dyn_model.n_u)
sym_um_car1 = ca.MX.sym('um_car1', car1_dyn_model.n_u)
sym_um_car2 = ca.MX.sym('um_car2', car2_dyn_model.n_u)
sym_um_car3 = ca.MX.sym('um_car3', car3_dyn_model.n_u)

car1_x_idx = 0
car1_y_idx = 1
car1_v_idx = 2
car1_p_idx = 3

car2_x_idx = 4
car2_y_idx = 5
car2_v_idx = 6
car2_p_idx = 7

car3_x_idx = 8
car3_y_idx = 9
car3_v_idx = 10
car3_p_idx = 11

ua_idx = 0
us_idx = 1

car1_q = sym_q[[car1_x_idx, car1_y_idx, car1_v_idx, car1_p_idx]]
car2_q = sym_q[[car2_x_idx, car2_y_idx, car2_v_idx, car2_p_idx]]
car3_q = sym_q[[car3_x_idx, car3_y_idx, car3_v_idx, car3_p_idx]]
car1_pos = sym_q[[car1_x_idx, car1_y_idx]]
car2_pos = sym_q[[car2_x_idx, car2_y_idx]]
car3_pos = sym_q[[car3_x_idx, car3_y_idx]]

# Build symbolic cost functions
# Car 1
car1_quad_input_cost = (1/2)*(0.1*sym_u_car1[ua_idx]**2 \
                         + 0.1*sym_u_car1[us_idx]**2)
Q = np.diag([1.0, 10.0, 1.0, 1.0])
car1_quad_state_cost = (1/2)*ca.bilin(Q, car1_q-car1_goal, car1_q-car1_goal)

car1_sym_stage = car1_quad_input_cost \
                + car1_quad_state_cost

car1_sym_term = 10*car1_quad_state_cost

car1_sym_costs = []
for k in range(N):
    car1_sym_costs.append(ca.Function(f'car1_stage_{k}', [sym_q, sym_u_car1, sym_um_car1], [car1_sym_stage],
                                [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'car1_stage_cost_{k}']))
car1_sym_costs.append(ca.Function('car1_term', [sym_q], [car1_sym_term],
                            [f'q_{N}'], ['car1_term_cost']))

# Car 2
car2_quad_input_cost = (1/2)*(0.1*sym_u_car2[ua_idx]**2 \
                         + 0.1*sym_u_car2[us_idx]**2)
Q = np.diag([1.0, 10.0, 1.0, 1.0])
car2_quad_state_cost = (1/2)*ca.bilin(Q, car2_q-car2_goal, car2_q-car2_goal)

car2_sym_stage = car2_quad_input_cost \
                + car2_quad_state_cost

car2_sym_term = 10*car2_quad_state_cost

car2_sym_costs = []
for k in range(N):
    car2_sym_costs.append(ca.Function(f'car2_stage_{k}', [sym_q, sym_u_car2, sym_um_car2], [car2_sym_stage],
                                [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'car2_stage_cost_{k}']))
car2_sym_costs.append(ca.Function('car2_term', [sym_q], [car2_sym_term],
                            [f'q_{N}'], ['car2_term_cost']))

# Car 3
car3_quad_input_cost = (1/2)*(0.1*sym_u_car3[ua_idx]**2 \
                         + 0.1*sym_u_car3[us_idx]**2)
Q = np.diag([1.0, 10.0, 1.0, 1.0])
car3_quad_state_cost = (1/2)*ca.bilin(Q, car3_q-car3_goal, car3_q-car3_goal)

car3_sym_stage = car3_quad_input_cost \
                + car3_quad_state_cost

car3_sym_term = 10*car3_quad_state_cost

car3_sym_costs = []
for k in range(N):
    car3_sym_costs.append(ca.Function(f'car3_stage_{k}', [sym_q, sym_u_car3, sym_um_car3], [car3_sym_stage],
                                [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'car3_stage_cost_{k}']))
car3_sym_costs.append(ca.Function('car3_term', [sym_q], [car3_sym_term],
                            [f'q_{N}'], ['car3_term_cost']))

sym_costs = [car1_sym_costs, car2_sym_costs, car3_sym_costs]

# Build symbolic constraints g_i(x, u, um) <= 0
obs_d12 = car1_r + car2_r
obs_d13 = car1_r + car3_r
obs_d23 = car2_r + car3_r
obs_avoid_constr = ca.vertcat((obs_d12)**2 - ca.bilin(ca.DM.eye(2), car1_pos - car2_pos,  car1_pos - car2_pos),
                              (obs_d13)**2 - ca.bilin(ca.DM.eye(2), car1_pos - car3_pos,  car1_pos - car3_pos),
                              (obs_d23)**2 - ca.bilin(ca.DM.eye(2), car2_pos - car3_pos,  car2_pos - car3_pos))

car1_lane = ca.vertcat(straight_lane_1(car1_pos), straight_lane_2(car1_pos))
car2_lane = ca.vertcat(straight_lane_1(car2_pos), straight_lane_2(car2_pos))
car3_lane = ca.vertcat(ramp_lane_1(car3_pos), ramp_lane_2(car3_pos))

# Car 1
car1_constr_stage = car1_lane
car1_constr_term = car1_lane

car1_constrs = []
for k in range(N):
    if car1_constr_stage is None:
        car1_constrs.append(None)
    else:
        car1_constrs.append(ca.Function(f'car1_constrs_{k}', [sym_q, sym_u_car1, sym_um_car1], [car1_constr_stage]))
if car1_constr_term is None:
    car1_constrs.append(None)
else:
    car1_constrs.append(ca.Function(f'car1_constrs_{N}', [sym_q], [car1_constr_term]))

# Car 2
car2_constr_stage = car2_lane
car2_constr_term = car2_lane

car2_constrs = []
for k in range(N):
    if car2_constr_stage is None:
        car2_constrs.append(None)
    else:
        car2_constrs.append(ca.Function(f'car2_constrs_{k}', [sym_q, sym_u_car2, sym_um_car2], [car2_constr_stage]))
if car2_constr_term is None:
    car2_constrs.append(None)
else:
    car2_constrs.append(ca.Function(f'car2_constrs_{N}', [sym_q], [car2_constr_term]))

# Car 3
car3_constr_stage = car3_lane
car3_constr_term = car3_lane

car3_constrs = []
for k in range(N):
    if car3_constr_stage is None:
        car3_constrs.append(None)
    else:
        car3_constrs.append(ca.Function(f'car3_constrs_{k}', [sym_q, sym_u_car3, sym_um_car3], [car3_constr_stage]))
if car3_constr_term is None:
    car3_constrs.append(None)
else:
    car3_constrs.append(ca.Function(f'car3_constrs_{N}', [sym_q], [car3_constr_term]))

# Shared
shared_constr_stage = obs_avoid_constr
shared_constr_term = obs_avoid_constr

shared_constrs = []
for k in range(N):
    u = ca.vertcat(sym_u_car1, sym_u_car2, sym_u_car3)
    um = ca.vertcat(sym_um_car1, sym_um_car2, sym_um_car3)
    if k == 0:
        shared_constrs.append(None)
    else:
        shared_constrs.append(ca.Function(f'shared_constrs_{k}', [sym_q, u, um], [shared_constr_stage]))
shared_constrs.append(ca.Function(f'shared_constrs_{N}', [sym_q], [shared_constr_term]))

agent_constrs = [car1_constrs, car2_constrs, car3_constrs]

# Combined agent and shared constraints
joint_constr_stage_0 = None
joint_constr_stage = ca.vertcat(car1_lane, car2_lane, car3_lane, obs_avoid_constr)
joint_constr_term = ca.vertcat(car1_lane, car2_lane, car3_lane, obs_avoid_constr)

u = ca.vertcat(sym_u_car1, sym_u_car2, sym_u_car3)
um = ca.vertcat(sym_um_car1, sym_um_car2, sym_um_car3)

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

dgsqp_controller = DGSQP(joint_model, 
                        sym_costs, 
                        agent_constrs,
                        shared_constrs,
                        {'ub': state_input_ub, 'lb': state_input_lb},
                        dgsqp_params,
                        xy_plot=plot_env)

# algames_controller = ALGAMES(joint_model, 
#                             sym_costs, 
#                             joint_constrs, 
#                             {'ub': state_input_ub, 'lb': state_input_lb},
#                             algames_params,
#                             xy_plot=plot_env)

rng = np.random.default_rng(seed=1)

# =============================================
# Run Monte Carlo
# =============================================
samples = 1000
i = 0
while i < samples:
    # Sample initial states
    x_nom = 0.0
    y_nom = 0.15
    v_nom = 0.3
    p_nom = 0.0
    x_rand = x_nom + 0.5*rng.random() - 0.25
    y_rand = y_nom + 0.1*rng.random() - 0.05
    v_rand = v_nom * (1 + 0.06*rng.random() - 0.03)
    p_rand = p_nom + (5*rng.random() - 2.5)*np.pi/180
    car1_sim_state = VehicleState(t=0.0, 
                                    x=Position(x=x_rand, y=y_rand), 
                                    e=OrientationEuler(psi=p_rand),
                                    v=BodyLinearVelocity(v_long=v_rand, v_tran=0),
                                    w=BodyAngularVelocity(w_psi=0))

    x_nom = 0.5
    y_nom = 0.15
    v_nom = 0.3
    p_nom = 0.0
    x_rand = x_nom + 0.5*rng.random() - 0.25
    y_rand = y_nom + 0.1*rng.random() - 0.05
    v_rand = v_nom * (1 + 0.06*rng.random() - 0.03)
    p_rand = p_nom + (5*rng.random() - 2.5)*np.pi/180
    car2_sim_state = VehicleState(t=0.0, 
                                    x=Position(x=x_rand, y=y_rand), 
                                    e=OrientationEuler(psi=p_rand),
                                    v=BodyLinearVelocity(v_long=v_rand, v_tran=0),
                                    w=BodyAngularVelocity(w_psi=0))

    # Car 3 is on ramp
    x_nom = 0.25
    y_nom = -(float(x7[0]+x5[0])/2-0.25)*np.tan(th)
    v_nom = 0.3
    p_nom = np.pi/12
    s_rand = 0.5*rng.random() - 0.25
    ey_rand = 0.1*rng.random() - 0.05
    x_rand = x_nom + s_rand*np.cos(th) - ey_rand*np.sin(th)
    y_rand = y_nom + s_rand*np.sin(th) + ey_rand*np.cos(th)
    v_rand = v_nom * (1 + 0.06*rng.random() - 0.03)
    p_rand = p_nom + (5*rng.random() - 2.5)*np.pi/180
    car3_sim_state = VehicleState(t=0.0, 
                                    x=Position(x=x_rand, y=y_rand), 
                                    e=OrientationEuler(psi=p_rand),
                                    v=BodyLinearVelocity(v_long=v_rand, v_tran=0),
                                    w=BodyAngularVelocity(w_psi=0))

    car1_init_state = copy.deepcopy(car1_sim_state)
    car2_init_state = copy.deepcopy(car2_sim_state)
    car3_init_state = copy.deepcopy(car3_sim_state)

    car1_u_ws = np.zeros((N, car1_dyn_model.n_u))
    car2_u_ws = np.zeros((N, car2_dyn_model.n_u))
    car3_u_ws = np.zeros((N, car3_dyn_model.n_u))

    car1_q_ws = np.zeros((N+1, car1_dyn_model.n_q))
    car2_q_ws = np.zeros((N+1, car2_dyn_model.n_q))
    car3_q_ws = np.zeros((N+1, car3_dyn_model.n_q))
    car1_q_ws[0] = car1_dyn_model.state2q(car1_init_state)
    car2_q_ws[0] = car1_dyn_model.state2q(car2_init_state)
    car2_q_ws[0] = car1_dyn_model.state2q(car2_init_state)
    for k in range(N):
        car1_q_ws[k+1] = np.array(car1_dyn_model.fd(car1_q_ws[k], car1_u_ws[k])).squeeze()
        car2_q_ws[k+1] = np.array(car2_dyn_model.fd(car2_q_ws[k], car2_u_ws[k])).squeeze()
        car3_q_ws[k+1] = np.array(car3_dyn_model.fd(car3_q_ws[k], car3_u_ws[k])).squeeze()

    if check_collision([car1_q_ws, car2_q_ws, car3_q_ws], [car1_r, car2_r, car3_r]):
        continue

    print(f'Sample {i+1}')

    # Initialize inputs
    car1_sim_state.u.u_a, car1_sim_state.u.u_steer = 0.0, 0.0
    car2_sim_state.u.u_a, car2_sim_state.u.u_steer = 0.0, 0.0
    car3_sim_state.u.u_a, car3_sim_state.u.u_steer = 0.0, 0.0
    joint_state = [car1_sim_state, car2_sim_state, car3_sim_state]

    # Solve with DG-SQP
    dgsqp_info = dgsqp_controller.solve(copy.deepcopy(joint_state))
    dgsqp_res = {'solve_info': copy.deepcopy(dgsqp_info), 
                    'params': copy.deepcopy(dgsqp_params), 
                    'init': copy.deepcopy(joint_state)}
    # preds = joint_model.qu2prediction(None, dgsqp_controller.q_pred, dgsqp_controller.u_pred)

    # Solve with ALGAMES
    # algames_info = algames_controller.solve(copy.deepcopy(joint_state))
    # algames_res = {'solve_info': copy.deepcopy(algames_info), 
    #                 'params': copy.deepcopy(algames_params), 
    #                 'init': copy.deepcopy(joint_state)}
    # preds = joint_model.qu2prediction(None, algames_controller.q_pred, algames_controller.u_pred)

    # Save results
    if save_data:
        # results = dict(dgsqp=dgsqp_res, algames=algames_res, env=merge_env)
        results = dict(dgsqp=dgsqp_res, env=merge_env)
        filename = f'sample_{i+1}.pkl'
        data_path = data_dir.joinpath(filename)
        with open(data_path, 'wb') as f:
            pickle.dump(results, f)

    i += 1

