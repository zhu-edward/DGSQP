from DGSQP.solvers.CA_LTV_MPC import CA_LTV_MPC
from DGSQP.solvers.solver_types import CALTVMPCParams

from DGSQP.types import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from DGSQP.dynamics.dynamics_models import CasadiDynamicCLBicycle
from DGSQP.dynamics.model_types import DynamicBicycleConfig

from DGSQP.tracks.track_lib import get_track, load_mpclab_raceline

from globals import dt, TRACK, M, DISCRETIZATION_METHOD

import numpy as np
import casadi as ca
import copy
import pdb

def get_warm_start_solver(args):
    track_constraints = True

    N = int(args.N)
    track = get_track(TRACK)
    L = track.track_length
    H = track.half_width

    path = f"{TRACK}_raceline.npz"
    raceline, s2t, raceline_mat = load_mpclab_raceline(path, TRACK, time_scale=1.7)

    t = 0

    car1_dynamics_config = DynamicBicycleConfig(dt=dt,
                                                model_name='dynamic_bicycle',
                                                noise=False,
                                                discretization_method=DISCRETIZATION_METHOD,
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
    car1_dyn_model = CasadiDynamicCLBicycle(t, car1_dynamics_config, track=track)

    car2_dynamics_config = DynamicBicycleConfig(dt=dt,
                                                model_name='dynamic_bicycle',
                                                noise=False,
                                                discretization_method=DISCRETIZATION_METHOD,
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
    car2_dyn_model = CasadiDynamicCLBicycle(t, car2_dynamics_config, track=track)

    if track_constraints:
        x_tran_lim = H
    else:
        x_tran_lim = 1e9

    state_input_ub = VehicleState(x=Position(x=1e9, y=1e9),
                                e=OrientationEuler(psi=1e9),
                                p=ParametricPose(s=2*L, x_tran=x_tran_lim, e_psi=1e9),
                                v=BodyLinearVelocity(v_long=5, v_tran=2),
                                w=BodyAngularVelocity(w_psi=10),
                                u=VehicleActuation(u_a=2.0, u_steer=0.436))
    state_input_lb = VehicleState(x=Position(x=-1e9, y=-1e9),
                                e=OrientationEuler(psi=-1e9),
                                p=ParametricPose(s=-2*L, x_tran=-x_tran_lim, e_psi=-1e9),
                                v=BodyLinearVelocity(v_long=0.1, v_tran=-2),
                                w=BodyAngularVelocity(w_psi=-10),
                                u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
    input_rate_ub = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
    input_rate_lb = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))
    
    # Symbolic placeholder variables
    sym_q = ca.SX.sym('q', car1_dyn_model.n_q)
    sym_u = ca.SX.sym('u', car1_dyn_model.n_u)
    sym_du = ca.SX.sym('du', car1_dyn_model.n_u)

    s_idx = 4
    ey_idx = 5

    ua_idx = 0
    us_idx = 1

    sym_q_ref = ca.SX.sym('q_ref', car1_dyn_model.n_q)

    Q = np.diag([1, 0, 0, 1, 1, 1])
    sym_state_stage = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)
    sym_state_term = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)

    sym_input_stage = 0.5*(1e-2*(sym_u[ua_idx])**2 + 1e-2*(sym_u[us_idx])**2)
    sym_input_term = 0.5*(1e-2*(sym_u[ua_idx])**2 + 1e-2*(sym_u[us_idx])**2)

    sym_rate_stage = 0.5*(0.1*(sym_du[ua_idx])**2 + 0.1*(sym_du[us_idx])**2)

    sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
    for k in range(N):
        sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q, sym_q_ref], [sym_state_stage])
        sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
        sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
    sym_costs['state'][N] = ca.Function('state_term', [sym_q, sym_q_ref], [sym_state_term])
    sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

    sym_constrs = {'state_input': [None for _ in range(N+1)], 
                    'rate': [None for _ in range(N)]}

    car1_mpc_params = CALTVMPCParams(N=N,
                                state_scaling=[10, 10, 7, np.pi/2, L, 1.0],
                                input_scaling=[2, 0.45],
                                damping=0.75,
                                qp_iters=5,
                                delay=None,
                                verbose=False,
                                qp_interface='casadi')

    car1_mpc_controller = CA_LTV_MPC(car1_dyn_model, 
                                sym_costs, 
                                sym_constrs, 
                                {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_ub, 'du_lb': input_rate_lb},
                                car1_mpc_params)
    
    car2_mpc_params = CALTVMPCParams(N=N,
                                state_scaling=[10, 10, 7, np.pi/2, L, 1.0],
                                input_scaling=[2, 0.45],
                                damping=0.75,
                                qp_iters=5,
                                delay=None,
                                verbose=False,
                                qp_interface='casadi')

    car2_mpc_controller = CA_LTV_MPC(car2_dyn_model, 
                                sym_costs, 
                                sym_constrs, 
                                {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_ub, 'du_lb': input_rate_lb},
                                car2_mpc_params)
    
    def warm_start_solver(joint_state):
        car1_state, car2_state = joint_state

        car1_lateral_offset = car1_state.p.x_tran - float(raceline(s2t(car1_state.p.s))[8])
        car1_t0 = float(s2t(car1_state.p.s))
        car1_scaling = car1_state.v.v_long/raceline(car1_t0)[3]
        t_ref = car1_t0 + dt*np.arange(N+1)*car1_scaling
        q_ref = np.array(raceline(t_ref)).squeeze().T[:,3:]
        q_ref[:,:3] = q_ref[:,:3]*car1_scaling
        q_ref[:,ey_idx] += car1_lateral_offset
        P = q_ref.ravel()

        u_ws = 0.01*np.ones((N+1, car1_dyn_model.n_u))
        du_ws = 0.01*np.ones((N, car1_dyn_model.n_u))

        car1_mpc_controller.set_warm_start(u_ws, du_ws, state=car1_state, parameters=P)
        success = car1_mpc_controller.solve(car1_state, parameters=P)

        if success:
            car1_u_ws = car1_mpc_controller.u_pred
            car1_ds = (car1_mpc_controller.q_pred[1:,s_idx] - car1_mpc_controller.q_pred[:-1,s_idx])/dt
            car1_pa_u_ws = np.hstack((car1_u_ws, car1_ds.reshape((-1,1))))
        else:
            return None

        car2_lateral_offset = car2_state.p.x_tran - float(raceline(s2t(car2_state.p.s))[8])
        car2_t0 = float(s2t(car2_state.p.s))
        car2_scaling = car2_state.v.v_long/raceline(car2_t0)[3]
        t_ref = car2_t0 + dt*np.arange(N+1)*car2_scaling
        q_ref = np.array(raceline(t_ref)).squeeze().T[:,3:]
        q_ref[:,:3] = q_ref[:,:3]*car2_scaling
        q_ref[:,ey_idx] += car2_lateral_offset
        P = q_ref.ravel()

        u_ws = 0.01*np.ones((N+1, car2_dyn_model.n_u))
        du_ws = 0.01*np.ones((N, car2_dyn_model.n_u))

        car2_mpc_controller.set_warm_start(u_ws, du_ws, state=car2_state, parameters=P)
        success = car2_mpc_controller.solve(car2_state, parameters=P)

        if success:
            car2_u_ws = car2_mpc_controller.u_pred
            car2_ds = (car2_mpc_controller.q_pred[1:,s_idx] - car2_mpc_controller.q_pred[:-1,s_idx])/dt
            car2_pa_u_ws = np.hstack((car2_u_ws, car2_ds.reshape((-1,1))))
        else:
            return None

        exact_ws = [car1_u_ws, car2_u_ws]
        approximate_ws = [car1_pa_u_ws, car2_pa_u_ws]

        if args.game_type == 'exact':
            return exact_ws
        elif args.game_type == 'approximate':
            return approximate_ws
    
    return warm_start_solver