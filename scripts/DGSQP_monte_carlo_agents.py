#!/usr/bin/env python3

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

# Initial time
t = 0

save_data = False

if save_data:
    time_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    data_dir = pathlib.Path(pathlib.Path.home(), f'results/sqgames_algams_mc_agents_{time_str}')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

# =============================================
# Helper functions
# =============================================
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

# Saturation cost fuction
sym_signed_u = ca.SX.sym('u', 1)
saturation_cost = ca.Function('saturation_cost', [sym_signed_u], [ca.fmax(ca.DM.zeros(1), sym_signed_u)])

dt = 0.1
discretization_method='euler'
half_width = 1.0

agent_dyn_config = KinematicBicycleConfig(dt=dt,
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

state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=np.pi))
state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-np.pi))

cost_params = dict(input_weight=[1.0, 1.0],
                         input_rate_weight=[1.0, 1.0],
                         comp_weights=[10.0, 5.0],
                         blocking_weight=0,
                         obs_weight=0,
                         obs_r=0.4)

use_ws=True
ibr_ws=False

# exp_M = [2, 3, 4]
# exp_N = [10, 15, 20]
# exp_theta = [15, 30, 45]
exp_M = [3]
exp_N = [25]
exp_theta = [90]
num_mc = 1
rng = np.random.default_rng()

for theta in exp_theta:
    track_obj = CurveTrack(enter_straight_length=1,
                            curve_length=8,
                            curve_swept_angle=theta*np.pi/180,
                            exit_straight_length=5,
                            width=half_width*2, 
                            slack=0.8,
                            ccw=True)
    for M in exp_M:
        agent_dyn_configs = []
        agent_dyn_models = []
        agent_xy_pos_idx = []
        agent_sey_pos_idx = []
        for i in range(M):
            agent_dyn_model = CasadiKinematicBicycleCombined(t, agent_dyn_config, track=track_obj)
            agent_dyn_configs.append(agent_dyn_config)
            agent_dyn_models.append(agent_dyn_model)
            agent_xy_pos_idx.append(np.array([0, 1])+int(np.sum([m.n_q for m in agent_dyn_models[:-1]])))
            agent_sey_pos_idx.append(np.array([4, 5])+int(np.sum([m.n_q for m in agent_dyn_models[:-1]])))

        # =============================================
        # Set up joint model
        # =============================================
        joint_model = CasadiDecoupledMultiAgentDynamicsModel(t, agent_dyn_models, joint_model_config)

        state_input_ub = [state_input_max for _ in range(M)]
        state_input_lb = [state_input_min for _ in range(M)]
        state_input_rate_ub = [state_input_rate_max for _ in range(M)]
        state_input_rate_lb = [state_input_rate_min for _ in range(M)]

        agent_cost_params = [cost_params for _ in range(M)]

        for N in exp_N:
            # =============================================
            # Solver setup
            # =============================================
            dgsqp_params = DGSQPParams(solver_name='DGSQP',
                                                dt=dt,
                                                N=N,
                                                reg=1e-3,
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

            agent_costs = []
            agent_constrs = []
            sym_q = ca.MX.sym('q', joint_model.n_q)
            for i in range(M):
                # Symbolic placeholder variables
                sym_u = ca.MX.sym('u', agent_dyn_models[i].n_u)
                sym_um = ca.MX.sym('um', agent_dyn_models[i].n_u)

                ua_idx = 0
                us_idx = 1

                pos = sym_q[agent_xy_pos_idx[i]]

                # Build symbolic cost functions
                quad_input_cost = (1/2)*(agent_cost_params[i]['input_weight'][0]*sym_u[ua_idx]**2 \
                                        + agent_cost_params[i]['input_weight'][1]*sym_u[us_idx]**2)
                quad_input_rate_cost = (1/2)*(agent_cost_params[i]['input_rate_weight'][0]*(sym_u[ua_idx]-sym_um[ua_idx])**2 \
                                            + agent_cost_params[i]['input_rate_weight'][1]*(sym_u[us_idx]-sym_um[us_idx])**2)
                # blocking_cost = (1/2)*agent_cost_params[i]['blocking_weight']*(sym_q[ego_ey_idx] - sym_q[tar_ey_idx])**2
                # obs_cost = (1/2)*agent_cost_params[i]['obs_weight']*saturation_cost(obs_cost_d-ca.norm_2(ego_pos - tar_pos))**2

                prog_cost = -agent_cost_params[i]['comp_weights'][0]*sym_q[agent_sey_pos_idx[i][0]]
                # ego_comp_cost = agent_cost_params['comp_weights'][1]*(sym_q[tar_s_idx]-sym_q[ego_s_idx])
                comp_cost = 0
                for j in range(M):
                    if j != i:
                        comp_cost += agent_cost_params[i]['comp_weights'][1]*ca.atan(sym_q[agent_sey_pos_idx[j][0]]-sym_q[agent_sey_pos_idx[i][0]])
            
                sym_stage = quad_input_cost \
                                + quad_input_rate_cost

                sym_term = prog_cost \
                                + comp_cost

                sym_costs = []
                for k in range(N):
                    sym_costs.append(ca.Function(f'agent_{i}_stage_{k}', [sym_q, sym_u, sym_um], [sym_stage]))
                sym_costs.append(ca.Function(f'agent_{i}_term', [sym_q], [sym_term]))

                agent_costs.append(sym_costs)

                # Build symbolic constraints g_i(x, u, um) <= 0
                input_rate_constr = ca.vertcat((sym_u[ua_idx]-sym_um[ua_idx]) - dt*state_input_rate_ub[i].u.u_a, 
                                                dt*state_input_rate_lb[i].u.u_a - (sym_u[ua_idx]-sym_um[ua_idx]),
                                                (sym_u[us_idx]-sym_um[us_idx]) - dt*state_input_rate_ub[i].u.u_steer, 
                                                dt*state_input_rate_lb[i].u.u_steer - (sym_u[us_idx]-sym_um[us_idx]))

                constr_stage = input_rate_constr
                constr_term = None

                constrs = []
                for k in range(N):
                    constrs.append(ca.Function(f'agent_{i}_constrs_{k}', [sym_q, sym_u, sym_um], [constr_stage]))
                if constr_term is None:
                    constrs.append(None)
                else:
                    constrs.append(ca.Function(f'agent_{i}_constrs_{N}', [sym_q], [constr_term]))
                
                agent_constrs.append(constrs)

            obs_avoid_constr = []
            for i in range(M):
                for j in range(i+1, M):
                    obs_d = agent_cost_params[i]['obs_r'] + agent_cost_params[j]['obs_r']
                    obs_avoid_constr.append((obs_d)**2 - ca.bilin(ca.DM.eye(2), sym_q[agent_xy_pos_idx[i]] - sym_q[agent_xy_pos_idx[j]],  sym_q[agent_xy_pos_idx[i]] - sym_q[agent_xy_pos_idx[j]]))
            obs_avoid_constr = ca.vertcat(*obs_avoid_constr)

            shared_constrs = []
            for k in range(N):
                if k == 0:
                    shared_constrs.append(None)
                else:
                    shared_constrs.append(ca.Function(f'shared_constrs_{k}', [sym_q, ca.MX.sym('u', joint_model.n_u), ca.MX.sym('um', joint_model.n_u)], [obs_avoid_constr]))
            shared_constrs.append(ca.Function(f'shared_constrs_{N}', [sym_q], [obs_avoid_constr]))
            dgsqp_solver = DGSQP(joint_model, 
                                agent_costs, 
                                agent_constrs,
                                shared_constrs,
                                {'ub': state_input_ub, 'lb': state_input_lb},
                                dgsqp_params)
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
                                            agent_costs, 
                                            agent_constrs,
                                            shared_constrs,
                                            {'ub': state_input_ub, 'lb': state_input_lb},
                                            ibr_params)

            first_seg_len = track_obj.cl_segs[0,0]
            sq_res = []
            for n in range(num_mc):
                print('========================================================')
                print(f'{M} agents, curved track with {theta} degree turn, control horizon: {N}, trial: {n+1}')
                while True:
                    joint_state = [VehicleState(t=t) for _ in range(M)]
                    
                    agent_q = []
                    agent_u = []
                    for i in range(M):
                        joint_state[i].p.s = max(0.1, rng.random()*first_seg_len)
                        joint_state[i].p.x_tran = rng.random()*half_width*2 - half_width
                        joint_state[i].v.v_long = rng.random()+2

                        track_obj.local_to_global_typed(joint_state[i])

                        if use_ws:
                            agent_steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                                                                x_ref=joint_state[i].p.x_tran,
                                                                u_max=state_input_ub[i].u.u_steer, 
                                                                u_min=state_input_lb[i].u.u_steer, 
                                                                du_max=state_input_rate_ub[i].u.u_steer, 
                                                                du_min=state_input_rate_lb[i].u.u_steer)
                            agent_speed_params = PIDParams(dt=dt, Kp=1.0,
                                                                x_ref=joint_state[i].v.v_long,
                                                                u_max=state_input_ub[i].u.u_a, 
                                                                u_min=state_input_lb[i].u.u_a, 
                                                                du_max=state_input_rate_ub[i].u.u_a, 
                                                                du_min=state_input_rate_lb[i].u.u_a)
                            pid_controller = PIDLaneFollower(dt, agent_steer_params, agent_speed_params)

                            # Construct initial guess for ALGAMES MPC with PID
                            agent_state = [copy.deepcopy(joint_state[i])]
                            for _ in range(N):
                                state = copy.deepcopy(agent_state[-1])
                                pid_controller.step(state)
                                agent_dyn_models[i].step(state)
                                agent_state.append(state)
                            
                            agent_q_ws = np.zeros((N+1, agent_dyn_models[i].n_q))
                            agent_u_ws = np.zeros((N, agent_dyn_models[i].n_u))
                            for k in range(N+1):
                                agent_q_ws[k] = np.array([agent_state[k].x.x, agent_state[k].x.y, agent_state[k].v.v_long, agent_state[k].p.e_psi, agent_state[k].p.s-1e-6, agent_state[k].p.x_tran])
                                if k < N:
                                    agent_u_ws[k] = np.array([agent_state[k+1].u.u_a, agent_state[k+1].u.u_steer])
                            agent_q.append(agent_q_ws)
                            agent_u.append(agent_u_ws)

                    collision = check_collision(agent_q, [agent_cost_params[i]['obs_r'] for i in range(M)])
                    if not collision:
                        break

                if ibr_ws:
                    ibr_controller.set_warm_start(agent_u)

                # =============================================
                # Run for a single step
                # =============================================
                # Initialize inputs
                for i in range(M):
                    joint_state[i].u.u_a, joint_state[i].u.u_steer = 0.0, 0.0

                if ibr_ws:
                    ibr_controller.step(copy.deepcopy(joint_state))
                    dgsqp_solver.set_warm_start(ibr_controller.u_pred)
                else:
                    dgsqp_solver.set_warm_start(np.hstack(agent_u))

                dgsqp_info = dgsqp_solver.solve(copy.deepcopy(joint_state))
                dgsqp_res = {'solve_info': copy.deepcopy(dgsqp_info), 
                                'params': copy.deepcopy(dgsqp_params), 
                                'init': copy.deepcopy(joint_state)}

                sq_res.append(dgsqp_res)

            results = dict(sqgames=sq_res,
                            track=track_obj, 
                            agent_dyn_configs=agent_dyn_configs,
                            joint_model_config=joint_model_config)

            if save_data:
                filename = f'data_c_{theta}_M_{M}_N_{N}.pkl'
                data_path = data_dir.joinpath(filename)
                with open(data_path, 'wb') as f:
                    pickle.dump(results, f)