#!/usr/bin/env python3

from DGSQP.solvers.CA_LTV_MPC import CA_LTV_MPC
from DGSQP.solvers.solver_types import CALTVMPCParams

from DGSQP.types import VehicleState, VehicleActuation, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from DGSQP.dynamics.dynamics_models import CasadiDynamicBicycleCombined
from DGSQP.dynamics.model_types import DynamicBicycleConfig
from DGSQP.tracks.track_lib import get_track

import numpy as np
import casadi as ca

track_obj = get_track('L_track_barc')
L = track_obj.track_length
H = track_obj.half_width

# =============================================
# Set up model
# =============================================
discretization_method = 'rk4'
dt = 0.1
dynamics_config = DynamicBicycleConfig(dt=dt,
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
dyn_model = CasadiDynamicBicycleCombined(0.0, dynamics_config, track=track_obj)

state_input_ub = VehicleState(x=Position(x=10, y=10),
                              p=ParametricPose(s=2*L, x_tran=(H), e_psi=100),
                              v=BodyLinearVelocity(v_long=10, v_tran=10),
                              w=BodyAngularVelocity(w_psi=10),
                              u=VehicleActuation(u_a=2.1, u_steer=0.436))
state_input_lb = VehicleState(x=Position(x=-10, y=-10),
                              p=ParametricPose(s=-2*L, x_tran=-(H), e_psi=-100),
                              v=BodyLinearVelocity(v_long=-10, v_tran=-10),
                              w=BodyAngularVelocity(w_psi=-10),
                              u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
input_rate_ub = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
input_rate_lb = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))

obs_r = 0.21

# =============================================
# MPC controller setup
# =============================================
N = 20
mpc_params = CALTVMPCParams(N=N,
                            # state_scaling=[4, 2, 7, 3, np.pi, 2*np.pi, 2*L, 1.5*H],
                            # input_scaling=[2.1, 0.436],
                            state_scaling=None,
                            input_scaling=None,
                            soft_state_bound_idxs=[7],
                            soft_state_bound_quad=[5],
                            soft_state_bound_lin=[25],
                            wrapped_state_idxs=[6],
                            wrapped_state_periods=[L],
                            damping=0.75,
                            qp_iters=2,
                            delay=None,
                            verbose=False,
                            qp_interface='casadi')

# Symbolic placeholder variables
sym_q = ca.MX.sym('q', dyn_model.n_q)
sym_u = ca.MX.sym('u', dyn_model.n_u)
sym_du = ca.MX.sym('du', dyn_model.n_u)

sym_q_ref = ca.MX.sym('q_ref', dyn_model.n_q)
sym_p_obs = ca.MX.sym('p_obs', 2)

x_idx = 0
y_idx = 1
s_idx = 6
ey_idx = 7

pos = sym_q[[x_idx, y_idx]]

ua_idx = 0
us_idx = 1

Q = np.diag([0, 0, 1, 0, 0, 1, 1, 1])
sym_state_stage = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)
sym_state_term = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)  - 1.0*sym_q[s_idx]

sym_input_stage = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
sym_input_term = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)

sym_rate_stage = 0.5*(0.01*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)

sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
for k in range(N):
    sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q, sym_q_ref], [sym_state_stage])
    sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
    sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
sym_costs['state'][N] = ca.Function('state_term', [sym_q, sym_q_ref], [sym_state_term])
sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

obs_avoid = ca.Function('obs_avoid', [sym_q, sym_u, sym_p_obs], [(2*obs_r)**2 - ca.bilin(ca.DM.eye(2), pos-sym_p_obs, pos-sym_p_obs)])
sym_constrs = {'state_input': [None] + [obs_avoid for _ in range(N)], 
                'rate': [None for _ in range(N)]}

mpc_controller = CA_LTV_MPC(dyn_model, 
                                sym_costs, 
                                sym_constrs, 
                                {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_ub, 'du_lb': input_rate_lb},
                                mpc_params)

