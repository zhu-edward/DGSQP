#!/usr/bin/env python3

from DGSQP.solvers.DGSQP import DGSQP
from DGSQP.solvers.PID import PIDLaneFollower
from DGSQP.solvers.solver_types import DGSQPParams, PIDParams

from DGSQP.types import VehicleState, VehicleActuation, Position, ParametricPose, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from DGSQP.dynamics.dynamics_models import CasadiKinematicUnicycleCombined, CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.dynamics.model_types import UnicycleConfig, MultiAgentModelConfig

from DGSQP.tracks.track_lib import get_track

import numpy as np
import casadi as ca

import copy

# Initial time
t = 0

dt = 0.1
N = 20

mu = 0.9
g = 9.81

track_obj = get_track('L_track_barc')
H = track_obj.half_width
L = track_obj.track_length

# =============================================
# Set up joint model
# =============================================
discretization_method='euler'
car1_dynamics_config = UnicycleConfig(dt=dt,
                                        model_name='car1_pm',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        drag_coefficient=0,
                                        damping_coefficient=0,
                                        rolling_resistance=0,
                                        code_gen=False,
                                        M=1)
car1_dyn_model = CasadiKinematicUnicycleCombined(t, car1_dynamics_config, track=track_obj)

car2_dynamics_config = UnicycleConfig(dt=dt,
                                        model_name='car2_pm',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        drag_coefficient=0,
                                        damping_coefficient=0,
                                        rolling_resistance=0,
                                        code_gen=False,
                                        M=1)
car2_dyn_model = CasadiKinematicUnicycleCombined(t, car2_dynamics_config, track=track_obj)

joint_config = MultiAgentModelConfig(dt=dt,
                                    discretization_method=discretization_method,
                                    use_mx=False,
                                    code_gen=False,
                                    verbose=True,
                                    compute_hessians=True,
                                    M=1)
joint_model = CasadiDecoupledMultiAgentDynamicsModel(t, [car1_dyn_model, car2_dyn_model], joint_config)

car1_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=H-0.1, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.0, u_steer=2.0))
car1_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-(H-0.1), e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.0, u_steer=-2.0))
car1_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=np.inf, u_steer=np.inf))
car1_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-np.inf, u_steer=-np.inf))

car2_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=H-0.1, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.0, u_steer=2.0))
car2_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-(H-0.1), e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.0, u_steer=-2.0))
car2_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=np.inf, u_steer=np.inf))
car2_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-np.inf, u_steer=-np.inf))

state_input_ub = [car1_state_input_max, car2_state_input_max]
state_input_lb = [car1_state_input_min, car2_state_input_min]

car1_cost_params = dict(input_weight=[0.1, 0.1],
                         input_rate_weight=[0.1, 0.1],
                         comp_weights=[1.0, 5.0])
car2_cost_params = dict(input_weight=[0.1, 0.1],
                        input_rate_weight=[0.1, 0.1],
                        comp_weights=[1.0, 5.0])

car1_r=0.21
car2_r=0.21

# =============================================
# DGSQP controller setup
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
                            pause_on_plot=True,
                            save_iter_data=True,
                            time_limit=None)

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

fx_idx = 0
wz_idx = 1

car1_pos = sym_q[[car1_x_idx, car1_y_idx]]
car2_pos = sym_q[[car2_x_idx, car2_y_idx]]

# Build symbolic cost functions
car1_quad_input_cost = (1/2)*(car1_cost_params['input_weight'][0]*sym_u_car1[fx_idx]**2 \
                         + car1_cost_params['input_weight'][1]*sym_u_car1[wz_idx]**2)
car1_quad_input_rate_cost = (1/2)*(car1_cost_params['input_rate_weight'][0]*(sym_u_car1[fx_idx]-sym_um_car1[fx_idx])**2 \
                              + car1_cost_params['input_rate_weight'][1]*(sym_u_car1[wz_idx]-sym_um_car1[wz_idx])**2)

car1_prog_cost = -car1_cost_params['comp_weights'][0]*sym_q[car1_s_idx]
# car1_comp_cost = car1_cost_params['comp_weights'][1]*(sym_q[car2_s_idx]-sym_q[car1_s_idx])
car1_comp_cost = car1_cost_params['comp_weights'][1]*ca.atan(sym_q[car2_s_idx]-sym_q[car1_s_idx])

car1_sym_stage = car1_quad_input_cost

car1_sym_term = car1_prog_cost \
                + car1_comp_cost

car1_sym_costs = []
for k in range(N):
    car1_sym_costs.append(ca.Function(f'car1_stage_{k}', [sym_q, sym_u_car1, sym_um_car1], [car1_sym_stage]))
car1_sym_costs.append(ca.Function('car1_term', [sym_q], [car1_sym_term]))

car2_quad_input_cost = (1/2)*(car2_cost_params['input_weight'][0]*sym_u_car2[fx_idx]**2 \
                         + car2_cost_params['input_weight'][1]*sym_u_car2[wz_idx]**2)
car2_quad_input_rate_cost = (1/2)*(car2_cost_params['input_rate_weight'][0]*(sym_u_car2[fx_idx]-sym_um_car2[fx_idx])**2 \
                              + car2_cost_params['input_rate_weight'][1]*(sym_u_car2[wz_idx]-sym_um_car2[wz_idx])**2)

car2_prog_cost = -car2_cost_params['comp_weights'][0]*sym_q[car2_s_idx]
# car2_comp_cost = car2_cost_params['comp_weights'][1]*(sym_q[car1_s_idx]-sym_q[car2_s_idx])
car2_comp_cost = car2_cost_params['comp_weights'][1]*ca.atan(sym_q[car1_s_idx]-sym_q[car2_s_idx])

car2_sym_stage = car2_quad_input_cost

car2_sym_term = car2_prog_cost \
                + car2_comp_cost

car2_sym_costs = []
for k in range(N):
    car2_sym_costs.append(ca.Function(f'car2_stage_{k}', [sym_q, sym_u_car2, sym_um_car2], [car2_sym_stage]))
car2_sym_costs.append(ca.Function('car2_term', [sym_q], [car2_sym_term]))

sym_costs = [car1_sym_costs, car2_sym_costs]

# Build symbolic constraints g_i(x, u, um) <= 0
car1_input_rate_constr = ca.vertcat((sym_u_car1[fx_idx]-sym_um_car1[fx_idx]) - dt*car1_state_input_rate_max.u.u_a, 
                                   dt*car1_state_input_rate_min.u.u_a - (sym_u_car1[fx_idx]-sym_um_car1[fx_idx]),
                                   (sym_u_car1[wz_idx]-sym_um_car1[wz_idx]) - dt*car1_state_input_rate_max.u.u_steer, 
                                   dt*car1_state_input_rate_min.u.u_steer - (sym_u_car1[wz_idx]-sym_um_car1[wz_idx]))

car2_input_rate_constr = ca.vertcat((sym_u_car2[fx_idx]-sym_um_car2[fx_idx]) - dt*car2_state_input_rate_max.u.u_a, 
                                   dt*car2_state_input_rate_min.u.u_a - (sym_u_car2[fx_idx]-sym_um_car2[fx_idx]),
                                   (sym_u_car2[wz_idx]-sym_um_car2[wz_idx]) - dt*car2_state_input_rate_max.u.u_steer, 
                                   dt*car2_state_input_rate_min.u.u_steer - (sym_u_car2[wz_idx]-sym_um_car2[wz_idx]))

car1_friction_limit = sym_u_car1[wz_idx]**2 + sym_u_car1[fx_idx]**2 - (mu*car1_dyn_model.m*9.81)**2
car2_friction_limit = sym_u_car2[wz_idx]**2 + sym_u_car2[fx_idx]**2 - (mu*car2_dyn_model.m*9.81)**2

obs_d = car1_r + car2_r
obs_avoid_constr = (obs_d)**2 - ca.bilin(ca.DM.eye(2), car1_pos - car2_pos,  car1_pos - car2_pos)

# car1_constr_stage = car1_input_rate_constr
# car1_constr_stage = car1_friction_limit
car1_constr_stage = None
car1_constr_term = None

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

# car2_constr_stage = car2_input_rate_constr
# car2_constr_stage = car2_friction_limit
car2_constr_stage = None
car2_constr_term = None

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

dgsqp_planner = DGSQP(joint_model, 
                        sym_costs, 
                        agent_constrs,
                        shared_constrs,
                        {'ub': state_input_ub, 'lb': state_input_lb},
                        dgsqp_params,
                        xy_plot=lambda ax: track_obj.plot_map(ax))

def get_pid_ws(dt, N, dyn_model, x_init, v_ref, x_ref, state_input_bounds, state_input_rate_bounds):
    steer_params = PIDParams(dt=dt, Kp=4.0, Ki=0.1,
                            x_ref=x_ref,
                            u_max=state_input_bounds[1].u.u_steer, 
                            u_min=state_input_bounds[0].u.u_steer, 
                            du_max=state_input_rate_bounds[1].u.u_steer, 
                            du_min=state_input_rate_bounds[0].u.u_steer)
    speed_params = PIDParams(dt=dt, Kp=1.0, 
                            x_ref=v_ref,
                            u_max=state_input_bounds[1].u.u_a, 
                            u_min=state_input_bounds[0].u.u_a, 
                            du_max=state_input_rate_bounds[1].u.u_a, 
                            du_min=state_input_rate_bounds[0].u.u_a)
    pid_controller = PIDLaneFollower(dt, steer_params, speed_params)

    state_seq = [copy.deepcopy(x_init)]
    for i in range(N):
        state = copy.deepcopy(state_seq[-1])
        pid_controller.step(state)
        dyn_model.step(state)
        state_seq.append(state)

    q_ws = np.zeros((N+1, dyn_model.n_q))
    u_ws = np.zeros((N, dyn_model.n_u))
    for i in range(N+1):
        q_ws[i] = np.array([state_seq[i].x.x, state_seq[i].x.y, state_seq[i].v.v_long, state_seq[i].p.e_psi, state_seq[i].p.s, state_seq[i].p.x_tran])
        if i < N:
            u_ws[i] = np.array([state_seq[i+1].u.u_a, state_seq[i+1].u.u_steer])

    return q_ws, u_ws

pid_ws_fns = [lambda N, x_init, v_ref, x_ref: get_pid_ws(dt, N, car1_dyn_model, x_init, v_ref, x_ref, [car1_state_input_min, car1_state_input_max], [car1_state_input_rate_min, car1_state_input_rate_max]),
              lambda N, x_init, v_ref, x_ref: get_pid_ws(dt, N, car2_dyn_model, x_init, v_ref, x_ref, [car2_state_input_min, car2_state_input_max], [car2_state_input_rate_min, car2_state_input_rate_max])]