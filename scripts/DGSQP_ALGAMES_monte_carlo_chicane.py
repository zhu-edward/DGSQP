#!/usr/bin/env python3

from DGSQP.solvers.IBR import IBR
from DGSQP.solvers.DGSQP import DGSQP
from DGSQP.solvers.ALGAMES import ALGAMES
from DGSQP.solvers.PID import PIDLaneFollower

from DGSQP.solvers.solver_types import IBRParams, DGSQPParams, ALGAMESParams, PIDParams

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

time_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
data_dir = pathlib.Path(pathlib.Path.home(), f'results/dgsqp_algams_mc_chicane_{time_str}')
if not data_dir.exists():
    data_dir.mkdir(parents=True)

# =============================================
# Helper functions
# =============================================
def check_collision(ego_traj, tar_traj, obs_d):
    for k in range(ego_traj.shape[0]):
        d = np.linalg.norm(ego_traj[k,:2] - tar_traj[k,:2], ord=2)
        if d < obs_d:
            return True
    return False

# Saturation cost fuction
sym_signed_u = ca.SX.sym('u', 1)
saturation_cost = ca.Function('saturation_cost', [sym_signed_u], [ca.fmax(ca.DM.zeros(1), sym_signed_u)])

dt = 0.1
discretization_method='euler'
half_width = 1.0

ego_dynamics_config = KinematicBicycleConfig(dt=dt,
                                            model_name='kinematic_bicycle_cl',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            wheel_dist_front=0.13,
                                            wheel_dist_rear=0.13,
                                            drag_coefficient=0.1,
                                            slip_coefficient=0.1,
                                            code_gen=False)

tar_dynamics_config = KinematicBicycleConfig(dt=dt,
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

ego_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
ego_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
ego_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=np.pi))
ego_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-np.pi))

tar_state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
tar_state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
tar_state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=np.pi))
tar_state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-np.pi))

state_input_ub = [ego_state_input_max, tar_state_input_max]
state_input_lb = [ego_state_input_min, tar_state_input_min]

ego_cost_params = dict(input_weight=[1.0, 1.0],
                         input_rate_weight=[1.0, 1.0],
                         comp_weights=[10.0, 5.0],
                         blocking_weight=0,
                         obs_weight=0,
                         obs_r=0.3)
tar_cost_params = dict(input_weight=[1.0, 1.0],
                        input_rate_weight=[1.0, 1.0],
                        comp_weights=[10.0, 5.0],
                        blocking_weight=0,
                        obs_weight=0,
                        obs_r=0.3)

ego_r=0.2
tar_r=0.2

use_ws=True
ibr_ws=False

exp_N = [10, 15, 20, 25]
exp_theta = np.arange(15, 91, 15)
# exp_N = [25]
# exp_theta = [90]
num_mc = 100
rng = np.random.default_rng()

for theta in exp_theta:
    track_obj = ChicaneTrack(enter_straight_length=1,
                            curve1_length=4,
                            curve1_swept_angle=theta*np.pi/180,
                            mid_straight_length=1,
                            exit_straight_length=5,
                            curve2_length=4,
                            curve2_swept_angle=theta*np.pi/180,
                            width=half_width*2, 
                            slack=0.8,
                            mirror=False)
    for N in exp_N:
        # =============================================
        # Set up joint model
        # =============================================
        ego_dyn_model = CasadiKinematicBicycleCombined(t, ego_dynamics_config, track=track_obj)
        tar_dyn_model = CasadiKinematicBicycleCombined(t, tar_dynamics_config, track=track_obj)
        joint_model = CasadiDecoupledMultiAgentDynamicsModel(t, [ego_dyn_model, tar_dyn_model], joint_model_config)

        # =============================================
        # Solver setup
        # =============================================
        dgsqp_params = DGSQPParams(solver_name='SQGAMES',
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
                                            code_gen=False,
                                            jit=False,
                                            opt_flag='O3',
                                            solver_dir=None,
                                            debug_plot=False,
                                            pause_on_plot=True)
        algames_params = ALGAMESParams(solver_name='ALGAMES',
                                            dt=dt,
                                            N=N,
                                            outer_iters=50,
                                            line_search_iters=50,
                                            line_search_tol=1e-6,
                                            newton_iters=50,
                                            newton_step_tol=1e-9,
                                            ineq_tol=1e-3,
                                            eq_tol=1e-3,
                                            opt_tol=1e-3,
                                            rho=1.0,
                                            gamma=10.0,
                                            rho_max=1e7,
                                            beta=0.01,
                                            tau=0.5,
                                            q_reg=1e-3,
                                            u_reg=1e-3,
                                            verbose=False,
                                            code_gen=False,
                                            jit=False,
                                            opt_flag='O3',
                                            solver_dir=None,
                                            debug_plot=False,
                                            pause_on_plot=False)

        # Symbolic placeholder variables
        sym_q = ca.MX.sym('q', joint_model.n_q)
        sym_u_ego = ca.MX.sym('u_ego', ego_dyn_model.n_u)
        sym_u_tar = ca.MX.sym('u_tar', tar_dyn_model.n_u)
        sym_um_ego = ca.MX.sym('um_ego', ego_dyn_model.n_u)
        sym_um_tar = ca.MX.sym('um_tar', tar_dyn_model.n_u)

        ego_x_idx = 0
        ego_y_idx = 1
        ego_s_idx = 4
        ego_ey_idx = 5

        tar_x_idx = 6
        tar_y_idx = 7
        tar_s_idx = 10
        tar_ey_idx = 11

        ua_idx = 0
        us_idx = 1

        ego_pos = sym_q[[ego_x_idx, ego_y_idx]]
        tar_pos = sym_q[[tar_x_idx, tar_y_idx]]

        obs_cost_d = ego_cost_params['obs_r'] + tar_cost_params['obs_r']

        # Build symbolic cost functions
        ego_quad_input_cost = (1/2)*(ego_cost_params['input_weight'][0]*sym_u_ego[ua_idx]**2 \
                                + ego_cost_params['input_weight'][1]*sym_u_ego[us_idx]**2)
        ego_quad_input_rate_cost = (1/2)*(ego_cost_params['input_rate_weight'][0]*(sym_u_ego[ua_idx]-sym_um_ego[ua_idx])**2 \
                                    + ego_cost_params['input_rate_weight'][1]*(sym_u_ego[us_idx]-sym_um_ego[us_idx])**2)
        ego_blocking_cost = (1/2)*ego_cost_params['blocking_weight']*(sym_q[ego_ey_idx] - sym_q[tar_ey_idx])**2
        ego_obs_cost = (1/2)*ego_cost_params['obs_weight']*saturation_cost(obs_cost_d-ca.norm_2(ego_pos - tar_pos))**2

        ego_prog_cost = -ego_cost_params['comp_weights'][0]*sym_q[ego_s_idx]
        # ego_comp_cost = ego_cost_params['comp_weights'][1]*(sym_q[tar_s_idx]-sym_q[ego_s_idx])
        ego_comp_cost = ego_cost_params['comp_weights'][1]*ca.atan(sym_q[tar_s_idx]-sym_q[ego_s_idx])

        ego_sym_stage = ego_quad_input_cost \
                        + ego_quad_input_rate_cost \
                        + ego_blocking_cost \
                        + ego_obs_cost

        ego_sym_term = ego_prog_cost \
                        + ego_comp_cost \
                        + ego_blocking_cost \
                        + ego_obs_cost

        ego_sym_costs = []
        for k in range(N):
            ego_sym_costs.append(ca.Function(f'ego_stage_{k}', [sym_q, sym_u_ego, sym_um_ego], [ego_sym_stage],
                                        [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'ego_stage_cost_{k}']))
        ego_sym_costs.append(ca.Function('ego_term', [sym_q], [ego_sym_term],
                                    [f'q_{N}'], ['ego_term_cost']))

        tar_quad_input_cost = (1/2)*(tar_cost_params['input_weight'][0]*sym_u_tar[ua_idx]**2 \
                                + tar_cost_params['input_weight'][1]*sym_u_tar[us_idx]**2)
        tar_quad_input_rate_cost = (1/2)*(tar_cost_params['input_rate_weight'][0]*(sym_u_tar[ua_idx]-sym_um_tar[ua_idx])**2 \
                                    + tar_cost_params['input_rate_weight'][1]*(sym_u_tar[us_idx]-sym_um_tar[us_idx])**2)
        tar_blocking_cost = (1/2)*tar_cost_params['blocking_weight']*(sym_q[ego_ey_idx] - sym_q[tar_ey_idx])**2
        tar_obs_cost = (1/2)*tar_cost_params['obs_weight']*saturation_cost(obs_cost_d-ca.norm_2(ego_pos - tar_pos))**2

        tar_prog_cost = -tar_cost_params['comp_weights'][0]*sym_q[tar_s_idx]
        # tar_comp_cost = tar_cost_params['comp_weights'][1]*(sym_q[ego_s_idx]-sym_q[tar_s_idx])
        tar_comp_cost = tar_cost_params['comp_weights'][1]*ca.atan(sym_q[ego_s_idx]-sym_q[tar_s_idx])

        tar_sym_stage = tar_quad_input_cost \
                        + tar_quad_input_rate_cost \
                        + tar_blocking_cost \
                        + tar_obs_cost

        tar_sym_term = tar_prog_cost \
                        + tar_comp_cost \
                        + tar_blocking_cost \
                        + tar_obs_cost

        tar_sym_costs = []
        for k in range(N):
            tar_sym_costs.append(ca.Function(f'tar_stage_{k}', [sym_q, sym_u_tar, sym_um_tar], [tar_sym_stage],
                                        [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'tar_stage_cost_{k}']))
        tar_sym_costs.append(ca.Function('tar_term', [sym_q], [tar_sym_term],
                                    [f'q_{N}'], ['tar_term_cost']))

        sym_costs = [ego_sym_costs, tar_sym_costs]

        # Build symbolic constraints g_i(x, u, um) <= 0
        ego_input_rate_constr = ca.vertcat((sym_u_ego[ua_idx]-sym_um_ego[ua_idx]) - dt*ego_state_input_rate_max.u.u_a, 
                                        dt*ego_state_input_rate_min.u.u_a - (sym_u_ego[ua_idx]-sym_um_ego[ua_idx]),
                                        (sym_u_ego[us_idx]-sym_um_ego[us_idx]) - dt*ego_state_input_rate_max.u.u_steer, 
                                        dt*ego_state_input_rate_min.u.u_steer - (sym_u_ego[us_idx]-sym_um_ego[us_idx]))

        tar_input_rate_constr = ca.vertcat((sym_u_tar[ua_idx]-sym_um_tar[ua_idx]) - dt*tar_state_input_rate_max.u.u_a, 
                                        dt*tar_state_input_rate_min.u.u_a - (sym_u_tar[ua_idx]-sym_um_tar[ua_idx]),
                                        (sym_u_tar[us_idx]-sym_um_tar[us_idx]) - dt*tar_state_input_rate_max.u.u_steer, 
                                        dt*tar_state_input_rate_min.u.u_steer - (sym_u_tar[us_idx]-sym_um_tar[us_idx]))

        obs_d = ego_r + tar_r
        obs_avoid_constr = (obs_d)**2 - ca.bilin(ca.DM.eye(2), ego_pos - tar_pos,  ego_pos - tar_pos)

        ego_constr_stage = ego_input_rate_constr
        ego_constr_term = None

        ego_constrs = []
        for k in range(N):
            ego_constrs.append(ca.Function(f'ego_constrs_{k}', [sym_q, sym_u_ego, sym_um_ego], [ego_constr_stage]))
        if ego_constr_term is None:
            ego_constrs.append(None)
        else:
            ego_constrs.append(ca.Function(f'ego_constrs_{N}', [sym_q], [ego_constr_term]))

        tar_constr_stage = tar_input_rate_constr
        tar_constr_term = None

        # constr_stage = obs_avoid_constr
        constr_stage = ca.vertcat(ego_input_rate_constr, tar_input_rate_constr, obs_avoid_constr)
        constr_term = obs_avoid_constr

        tar_constrs = []
        for k in range(N):
            tar_constrs.append(ca.Function(f'tar_constrs_{k}', [sym_q, sym_u_tar, sym_um_tar], [tar_constr_stage]))
        if tar_constr_term is None:
            tar_constrs.append(None)
        else:
            tar_constrs.append(ca.Function(f'tar_constrs_{N}', [sym_q], [tar_constr_term]))

        shared_constr_stage = obs_avoid_constr
        shared_constr_term = obs_avoid_constr

        shared_constrs = []
        for k in range(N):
            if k == 0:
                shared_constrs.append(None)
            else:
                shared_constrs.append(ca.Function(f'shared_constrs_{k}', [sym_q, ca.vertcat(sym_u_ego, sym_u_tar), ca.vertcat(sym_um_ego, sym_um_tar)], [shared_constr_stage]))
        shared_constrs.append(ca.Function(f'shared_constrs_{N}', [sym_q], [shared_constr_term]))

        agent_constrs = [ego_constrs, tar_constrs]

        dgsqp_solver = DGSQP(joint_model, 
                            sym_costs, 
                            agent_constrs,
                            shared_constrs,
                            {'ub': state_input_ub, 'lb': state_input_lb},
                            dgsqp_params)

        joint_constr_stage_0 = ca.vertcat(ego_input_rate_constr, tar_input_rate_constr)
        joint_constr_stage = ca.vertcat(ego_input_rate_constr, tar_input_rate_constr, obs_avoid_constr)
        joint_constr_term = obs_avoid_constr

        joint_constrs = []
        for k in range(N):
            if k == 0:
                joint_constrs.append(ca.Function(f'nl_constrs_{k}', [sym_q, ca.vertcat(sym_u_ego, sym_u_tar), ca.vertcat(sym_um_ego, sym_um_tar)], [joint_constr_stage_0]))
            else:
                joint_constrs.append(ca.Function(f'nl_constrs_{k}', [sym_q, ca.vertcat(sym_u_ego, sym_u_tar), ca.vertcat(sym_um_ego, sym_um_tar)], [joint_constr_stage]))
        joint_constrs.append(ca.Function(f'nl_constrs_{N}', [sym_q], [joint_constr_term]))

        algames_solver = ALGAMES(joint_model, 
                                sym_costs, 
                                joint_constrs, 
                                {'ub': state_input_ub, 'lb': state_input_lb},
                                algames_params)

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
                                    code_gen=False,
                                    jit=False,
                                    opt_flag='O3',
                                    solver_dir=None,
                                    debug_plot=False,
                                    pause_on_plot=True)
            ibr_solver = IBR(joint_model, 
                            sym_costs, 
                            agent_constrs,
                            shared_constrs,
                            {'ub': state_input_ub, 'lb': state_input_lb},
                            ibr_params)

        first_seg_len = track_obj.cl_segs[0,0]
        sq_res = []
        al_res = []
        for i in range(num_mc):
            print('========================================================')
            print(f'Curved track with {theta} degree turn, control horizon: {N}, trial: {i+1}')
            while True:
                ego_sim_state = VehicleState(t=t)
                tar_sim_state = VehicleState(t=t)

                ego_sim_state.p.s = max(0.1, rng.random()*first_seg_len)
                ego_sim_state.p.x_tran = rng.random()*half_width*2 - half_width
                ego_sim_state.v.v_long = rng.random()+2

                d = 2*np.pi*rng.random()
                tar_sim_state.p.s = ego_sim_state.p.s + 1.2*obs_d*np.cos(d)
                if tar_sim_state.p.s < 0:
                    continue
                tar_sim_state.p.x_tran = ego_sim_state.p.x_tran + 1.2*obs_d*np.sin(d)
                if np.abs(tar_sim_state.p.x_tran) > half_width:
                    continue
                # tar_sim_state.p.s = rng.random()*first_seg_len/2
                # tar_sim_state.p.x_tran = rng.random()*half_width*2 - half_width
                tar_sim_state.v.v_long = rng.random()+2

                track_obj.local_to_global_typed(ego_sim_state)
                track_obj.local_to_global_typed(tar_sim_state)

                # =============================================
                # Warm start controller setup
                # =============================================
                if use_ws:
                    # Set up PID controllers for warm start
                    ego_steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                                                u_max=ego_state_input_max.u.u_steer, 
                                                u_min=ego_state_input_min.u.u_steer, 
                                                du_max=ego_state_input_rate_max.u.u_steer, 
                                                du_min=ego_state_input_rate_min.u.u_steer)
                    ego_speed_params = PIDParams(dt=dt, Kp=1.0, 
                                                u_max=ego_state_input_max.u.u_a, 
                                                u_min=ego_state_input_min.u.u_a, 
                                                du_max=ego_state_input_rate_max.u.u_a, 
                                                du_min=ego_state_input_rate_min.u.u_a)
                    ego_v_ref = ego_sim_state.v.v_long
                    # ego_v_ref = tar_sim_state.v.v_long
                    ego_x_ref = ego_sim_state.p.x_tran
                    ego_pid_controller = PIDLaneFollower(ego_v_ref, ego_x_ref, dt, ego_steer_params, ego_speed_params)

                    tar_steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                                                u_max=tar_state_input_max.u.u_steer, 
                                                u_min=tar_state_input_min.u.u_steer, 
                                                du_max=tar_state_input_rate_max.u.u_steer, 
                                                du_min=tar_state_input_rate_min.u.u_steer)
                    tar_speed_params = PIDParams(dt=dt, Kp=1.0, 
                                                u_max=tar_state_input_max.u.u_a, 
                                                u_min=tar_state_input_min.u.u_a, 
                                                du_max=tar_state_input_rate_max.u.u_a, 
                                                du_min=tar_state_input_rate_min.u.u_a)
                    tar_v_ref = tar_sim_state.v.v_long
                    tar_x_ref = tar_sim_state.p.x_tran
                    # tar_x_ref = ego_sim_state.p.x_tran
                    tar_pid_controller = PIDLaneFollower(tar_v_ref, tar_x_ref, dt, tar_steer_params, tar_speed_params)

                    # Construct initial guess for ALGAMES MPC with PID
                    ego_state = [copy.deepcopy(ego_sim_state)]
                    for i in range(N):
                        state = copy.deepcopy(ego_state[-1])
                        ego_pid_controller.step(state)
                        ego_dyn_model.step(state)
                        ego_state.append(state)
                        
                    tar_state = [copy.deepcopy(tar_sim_state)]
                    for i in range(N):
                        state = copy.deepcopy(tar_state[-1])
                        tar_pid_controller.step(state)
                        tar_dyn_model.step(state)
                        tar_state.append(state)
                    
                    ego_q_ws = np.zeros((N+1, ego_dyn_model.n_q))
                    tar_q_ws = np.zeros((N+1, tar_dyn_model.n_q))
                    ego_u_ws = np.zeros((N, ego_dyn_model.n_u))
                    tar_u_ws = np.zeros((N, tar_dyn_model.n_u))
                    for i in range(N+1):
                        ego_q_ws[i] = np.array([ego_state[i].x.x, ego_state[i].x.y, ego_state[i].v.v_long, ego_state[i].p.e_psi, ego_state[i].p.s-1e-6, ego_state[i].p.x_tran])
                        tar_q_ws[i] = np.array([tar_state[i].x.x, tar_state[i].x.y, tar_state[i].v.v_long, tar_state[i].p.e_psi, tar_state[i].p.s-1e-6, tar_state[i].p.x_tran])
                        if i < N:
                            ego_u_ws[i] = np.array([ego_state[i+1].u.u_a, ego_state[i+1].u.u_steer])
                            tar_u_ws[i] = np.array([tar_state[i+1].u.u_a, tar_state[i+1].u.u_steer])

                    collision = check_collision(ego_q_ws, tar_q_ws, obs_d)
                    if not collision:
                        break

                if ibr_ws:
                    ibr_solver.set_warm_start([ego_u_ws, tar_u_ws])

            # =============================================
            # Run for a single step
            # =============================================
            # Initialize inputs
            ego_sim_state.u.u_a, ego_sim_state.u.u_steer = 0.0, 0.0
            tar_sim_state.u.u_a, tar_sim_state.u.u_steer = 0.0, 0.0
            joint_state = [ego_sim_state, tar_sim_state]

            if ibr_ws:
                ibr_solver.step(copy.deepcopy(joint_state))
                dgsqp_solver.set_warm_start(ibr_solver.u_pred)
                algames_solver.set_warm_start(ibr_solver.q_pred, ibr_solver.u_pred)
            else:
                dgsqp_solver.set_warm_start(np.hstack([ego_u_ws, tar_u_ws]))
                algames_solver.set_warm_start(np.hstack([ego_q_ws, tar_q_ws]), np.hstack([ego_u_ws, tar_u_ws]))

            dgsqp_info = dgsqp_solver.solve(copy.deepcopy(joint_state))
            dgsqp_res = {'solve_info': copy.deepcopy(dgsqp_info), 
                            'params': copy.deepcopy(dgsqp_params), 
                            'init': copy.deepcopy(joint_state)}

            algames_info = algames_solver.solve(copy.deepcopy(joint_state))
            algames_res = {'solve_info': copy.deepcopy(algames_info), 
                            'params': copy.deepcopy(algames_params), 
                            'init': copy.deepcopy(joint_state)}

            sq_res.append(dgsqp_res)
            al_res.append(algames_res)

        results = dict(dgsqp=sq_res, 
                        algames=al_res, 
                        track=track_obj, 
                        agent_dyn_configs=[ego_dynamics_config, tar_dynamics_config],
                        joint_model_config=joint_model_config)

        filename = f'data_c_{theta}_N_{N}.pkl'
        data_path = data_dir.joinpath(filename)
        with open(data_path, 'wb') as f:
            pickle.dump(results, f)