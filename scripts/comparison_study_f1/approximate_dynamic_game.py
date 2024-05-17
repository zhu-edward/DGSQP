#!/usr/bin/env python3

from DGSQP.types import VehicleState, VehicleActuation, Position, ParametricPose, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from DGSQP.dynamics.dynamics_models import CasadiDynamicBicycleProgressAugmented, CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.dynamics.model_types import DynamicBicycleConfig, MultiAgentModelConfig
from DGSQP.tracks.track_lib import get_track

import numpy as np
import casadi as ca

from globals import TRACK, dt, VL, VW, DISCRETIZATION_METHOD, M

def get_approximate_dynamic_game(N):
    # Initial time
    t = 0

    track_obj = get_track(TRACK).get_track_segment([60, 80], 10)

    # =============================================
    # Set up joint model
    # =============================================
    car1_dynamics_config = DynamicBicycleConfig(dt=dt,
                                                model_name='dynamic_bicycle',
                                                discretization_method=DISCRETIZATION_METHOD,
                                                wheel_dist_front=0.18,
                                                wheel_dist_rear=0.18,
                                                simple_slip=False,
                                                tire_model='pacejka',
                                                mass=70.0,
                                                yaw_inertia=2.0627,
                                                wheel_friction=0.9,
                                                pacejka_b_front=10.0,
                                                pacejka_b_rear=10.0,
                                                pacejka_c_front=2.5,
                                                pacejka_c_rear=2.5,
                                                M=M)
    car1_pa_dyn_model = CasadiDynamicBicycleProgressAugmented(t, car1_dynamics_config, track=track_obj)

    car2_dynamics_config = DynamicBicycleConfig(dt=dt,
                                                model_name='dynamic_bicycle',
                                                discretization_method=DISCRETIZATION_METHOD,
                                                wheel_dist_front=0.18,
                                                wheel_dist_rear=0.18,
                                                simple_slip=False,
                                                tire_model='pacejka',
                                                mass=70.0,
                                                yaw_inertia=2.0627,
                                                wheel_friction=0.9,
                                                pacejka_b_front=10.0,
                                                pacejka_b_rear=10.0,
                                                pacejka_c_front=2.5,
                                                pacejka_c_rear=2.5,
                                                M=M)
    car2_pa_dyn_model = CasadiDynamicBicycleProgressAugmented(t, car2_dynamics_config, track=track_obj)

    joint_config = MultiAgentModelConfig(dt=dt,
                                        discretization_method=DISCRETIZATION_METHOD,
                                        use_mx=False,
                                        code_gen=False,
                                        verbose=False,
                                        compute_hessians=True,
                                        M=M)
    joint_pa_model = CasadiDecoupledMultiAgentDynamicsModel(t, [car1_pa_dyn_model, car2_pa_dyn_model], joint_config)

    car1_state_input_max = VehicleState(x=Position(x=np.inf, y=np.inf),
                                p=ParametricPose(s=np.inf),
                                e=OrientationEuler(psi=np.inf),
                                v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                                w=BodyAngularVelocity(w_psi=np.inf),
                                u=VehicleActuation(u_a=2.1, u_steer=0.436, u_ds=4.0))
    car1_state_input_min = VehicleState(x=Position(x=-np.inf, y=-np.inf),
                                p=ParametricPose(s=-np.inf),
                                e=OrientationEuler(psi=-np.inf),
                                v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                                w=BodyAngularVelocity(w_psi=-np.inf),
                                u=VehicleActuation(u_a=-2.1, u_steer=-0.436, u_ds=0.0))
    car1_state_input_rate_max = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5, u_ds=5.0))
    car1_state_input_rate_min = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5, u_ds=-5.0))

    car2_state_input_max = VehicleState(x=Position(x=np.inf, y=np.inf),
                                p=ParametricPose(s=np.inf),
                                e=OrientationEuler(psi=np.inf),
                                v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                                w=BodyAngularVelocity(w_psi=np.inf),
                                u=VehicleActuation(u_a=2.1, u_steer=0.436, u_ds=4.0))
    car2_state_input_min = VehicleState(x=Position(x=-np.inf, y=-np.inf),
                                p=ParametricPose(s=-np.inf),
                                e=OrientationEuler(psi=-np.inf),
                                v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                                w=BodyAngularVelocity(w_psi=-np.inf),
                                u=VehicleActuation(u_a=-2.1, u_steer=-0.436, u_ds=0.0))
    car2_state_input_rate_max = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5, u_ds=5.0))
    car2_state_input_rate_min = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5, u_ds=-5.0))
    
    state_input_ub = [car1_state_input_max, car2_state_input_max]
    state_input_lb = [car1_state_input_min, car2_state_input_min]

    state_input_rate_ub = [car1_state_input_rate_max, car2_state_input_rate_max]
    state_input_rate_lb = [car1_state_input_rate_min, car2_state_input_rate_min]

    car1_cost_params = dict(input_weight=[1.0, 1.0, 1e-4],
                            input_rate_weight=[1.0, 1.0, 1e-4],
                            comp_weights=[1.0, 5.0])
    car2_cost_params = dict(input_weight=[1.0, 1.0, 1e-4],
                            input_rate_weight=[1.0, 1.0, 1e-4],
                            comp_weights=[1.0, 5.0])

    # =============================================
    # Solver setup with progres-augmented global-frame dynamics
    # =============================================

    # Symbolic placeholder variables
    sym_q = ca.SX.sym('q', joint_pa_model.n_q)
    sym_u_car1 = ca.SX.sym('u_car1', car1_pa_dyn_model.n_u)
    sym_u_car2 = ca.SX.sym('u_car2', car2_pa_dyn_model.n_u)
    sym_um_car1 = ca.SX.sym('um_car1', car1_pa_dyn_model.n_u)
    sym_um_car2 = ca.SX.sym('um_car2', car2_pa_dyn_model.n_u)

    car1_x_idx = 3
    car1_y_idx = 4
    car1_p_idx = 5
    car1_s_idx = 6

    car2_x_idx = 10
    car2_y_idx = 11
    car2_p_idx = 12
    car2_s_idx = 13

    ua_idx = 0
    us_idx = 1
    uds_idx = 2

    car1_pos = sym_q[[car1_x_idx, car1_y_idx]]
    car2_pos = sym_q[[car2_x_idx, car2_y_idx]]

    # Build symbolic cost functions
    car1_quad_input_cost = (1/2)*(car1_cost_params['input_weight'][0]*sym_u_car1[ua_idx]**2 \
                                + car1_cost_params['input_weight'][1]*sym_u_car1[us_idx]**2 \
                                + car1_cost_params['input_weight'][2]*sym_u_car1[uds_idx]**2)
    car1_quad_input_rate_cost = (1/2)*(car1_cost_params['input_rate_weight'][0]*(sym_u_car1[ua_idx]-sym_um_car1[ua_idx])**2 \
                                    + car1_cost_params['input_rate_weight'][1]*(sym_u_car1[us_idx]-sym_um_car1[us_idx])**2 \
                                    + car1_cost_params['input_rate_weight'][2]*(sym_u_car1[uds_idx]-sym_um_car1[uds_idx])**2)

    car1_prog_cost = -car1_cost_params['comp_weights'][0]*sym_q[car1_s_idx]
    car1_comp_cost = car1_cost_params['comp_weights'][1]*(sym_q[car2_s_idx]-sym_q[car1_s_idx])

    car1_sym_stage = car1_quad_input_cost \
                    + car1_quad_input_rate_cost
    car1_sym_term = car1_prog_cost \
                    + car1_comp_cost            

    car1_sym_costs = []
    for k in range(N):
        car1_sym_costs.append(ca.Function(f'car1_stage_{k}', [sym_q, sym_u_car1, sym_um_car1], [car1_sym_stage],
                                    [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'car1_stage_cost_{k}']))
    car1_sym_costs.append(ca.Function('car1_term', [sym_q], [car1_sym_term],
                                [f'q_{N}'], ['car1_term_cost']))
                                
    car2_quad_input_cost = (1/2)*(car2_cost_params['input_weight'][0]*sym_u_car2[ua_idx]**2 \
                                + car2_cost_params['input_weight'][1]*sym_u_car2[us_idx]**2 \
                                + car2_cost_params['input_weight'][2]*sym_u_car2[uds_idx]**2)
    car2_quad_input_rate_cost = (1/2)*(car2_cost_params['input_rate_weight'][0]*(sym_u_car2[ua_idx]-sym_um_car2[ua_idx])**2 \
                                    + car2_cost_params['input_rate_weight'][1]*(sym_u_car2[us_idx]-sym_um_car2[us_idx])**2 \
                                    + car2_cost_params['input_rate_weight'][2]*(sym_u_car2[uds_idx]-sym_um_car2[uds_idx])**2)

    car2_prog_cost = -car2_cost_params['comp_weights'][0]*sym_q[car2_s_idx]
    car2_comp_cost = car2_cost_params['comp_weights'][1]*(sym_q[car1_s_idx]-sym_q[car2_s_idx])

    car2_sym_stage = car2_quad_input_cost \
                    + car2_quad_input_rate_cost

    car2_sym_term = car2_prog_cost \
                    + car2_comp_cost

    car2_sym_costs = []
    for k in range(N):
        car2_sym_costs.append(ca.Function(f'car2_stage_{k}', [sym_q, sym_u_car2, sym_um_car2], [car2_sym_stage],
                                    [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'car2_stage_cost_{k}']))
    car2_sym_costs.append(ca.Function('car2_term', [sym_q], [car2_sym_term],
                                [f'q_{N}'], ['car2_term_cost']))

    agent_costs = [car1_sym_costs, car2_sym_costs]

    # Build symbolic constraints g_i(x, u, um) <= 0
    car1_input_rate_constr = ca.vertcat((sym_u_car1[ua_idx]-sym_um_car1[ua_idx]) - dt*car1_state_input_rate_max.u.u_a, 
                                    dt*car1_state_input_rate_min.u.u_a - (sym_u_car1[ua_idx]-sym_um_car1[ua_idx]),
                                    (sym_u_car1[us_idx]-sym_um_car1[us_idx]) - dt*car1_state_input_rate_max.u.u_steer, 
                                    dt*car1_state_input_rate_min.u.u_steer - (sym_u_car1[us_idx]-sym_um_car1[us_idx]))

    car2_input_rate_constr = ca.vertcat((sym_u_car2[ua_idx]-sym_um_car2[ua_idx]) - dt*car2_state_input_rate_max.u.u_a, 
                                    dt*car2_state_input_rate_min.u.u_a - (sym_u_car2[ua_idx]-sym_um_car2[ua_idx]),
                                    (sym_u_car2[us_idx]-sym_um_car2[us_idx]) - dt*car2_state_input_rate_max.u.u_steer, 
                                    dt*car2_state_input_rate_min.u.u_steer - (sym_u_car2[us_idx]-sym_um_car2[us_idx]))

    car1_obs_r = np.sqrt((VL/2)**2 + (VW/2)**2)
    car2_obs_r = np.sqrt((VL/2)**2 + (VW/2)**2)
    
    obs_avoid_constr = (car1_obs_r+car2_obs_r)**2 - ca.bilin(ca.DM.eye(2), car1_pos - car2_pos,  car1_pos - car2_pos)

    car1_constr_stage = None
    car1_constr_term = None
    car1_constrs = [None for _ in range(N+1)]

    car2_constr_stage = None
    car2_constr_term = None
    car2_constrs = [None for _ in range(N+1)]
    
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

    return N, joint_pa_model, agent_costs, agent_constrs, shared_constrs, state_input_lb, state_input_ub