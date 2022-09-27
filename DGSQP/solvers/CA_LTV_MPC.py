#!/usr/bin python3

import numpy as np
import scipy as sp

import casadi as ca
import cvxpy as cp

import time, copy

from typing import List, Dict
from collections import deque

from DGSQP.dynamics.dynamics_models import CasadiDynamicsModel
from DGSQP.types import VehicleState, VehiclePrediction

from DGSQP.solvers.abstract_solver import AbstractSolver
from DGSQP.solvers.solver_types import CALTVMPCParams

class CA_LTV_MPC(AbstractSolver):
    def __init__(self, dynamics: CasadiDynamicsModel, 
                       costs: Dict[str, List[ca.Function]], 
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState],
                       control_params: CALTVMPCParams=CALTVMPCParams(),
                       qp_interface='casadi',
                       print_method=print):

        self.dynamics       = dynamics
        self.dt             = dynamics.dt
        self.costs          = costs
        self.constraints    = constraints
        self.qp_interface   = qp_interface
        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method

        self.n_u            = self.dynamics.n_u
        self.n_q            = self.dynamics.n_q
        self.n_z            = self.n_q + self.n_u

        self.N = control_params.N

        if control_params.state_scaling:
            self.state_scaling = 1/np.array(control_params.state_scaling)
        else:
            self.state_scaling = np.ones(self.dynamics.n_q)
        if control_params.input_scaling:
            self.input_scaling  = 1/np.array(control_params.input_scaling)
        else:
            self.input_scaling = np.ones(self.dynamics.n_u)

        self.soft_state_bound_idxs      = control_params.soft_state_bound_idxs
        if self.soft_state_bound_idxs is not None:
            self.soft_state_bound_quad      = np.array(control_params.soft_state_bound_quad)
            self.soft_state_bound_lin       = np.array(control_params.soft_state_bound_lin)

        self.wrapped_state_idxs     = control_params.wrapped_state_idxs
        self.wrapped_state_periods  = control_params.wrapped_state_periods

        self.damping        = control_params.damping
        self.qp_iters       = control_params.qp_iters

        self.delay          = control_params.delay
        self.delay_buffer   = []
        if self.delay is None:
            self.delay = np.zeros(self.dynamics.n_u)
            self.delay_buffer = None

        self.solver_name    = control_params.solver_name
        self.verbose        = control_params.verbose
        self.code_gen       = control_params.code_gen
        self.jit            = control_params.jit
        self.opt_flag       = control_params.opt_flag

        if control_params.state_scaling:
            self.q_scaling = 1/np.array(control_params.state_scaling)
        else:
            self.q_scaling = np.ones(self.n_q)
        self.q_scaling_inv = 1/self.q_scaling
        if control_params.input_scaling:
            self.u_scaling  = 1/np.array(control_params.input_scaling)
        else:
            self.u_scaling = np.ones(self.n_u)
        self.u_scaling_inv = 1/self.u_scaling

        # Process box constraints
        self.state_ub, self.input_ub = self.dynamics.state2qu(bounds['qu_ub'])
        self.state_lb, self.input_lb = self.dynamics.state2qu(bounds['qu_lb'])
        _, self.input_rate_ub = self.dynamics.state2qu(bounds['du_ub'])
        _, self.input_rate_lb = self.dynamics.state2qu(bounds['du_lb'])

        self.qu_ub = np.concatenate((self.state_ub, self.input_ub))
        self.qu_lb = np.concatenate((self.state_lb, self.input_lb))
        self.D_lb = np.concatenate((np.tile(self.qu_lb, self.N+1), np.tile(self.input_rate_lb, self.N)))
        self.D_ub = np.concatenate((np.tile(self.qu_ub, self.N+1), np.tile(self.input_rate_ub, self.N)))

        self.n_c = [0 for _ in range(self.N+1)]

        # Construct normalization matrix
        self.qu_scaling     = np.concatenate((self.q_scaling, self.u_scaling))
        self.qu_scaling_inv = 1/self.qu_scaling
        
        self.D_scaling = np.concatenate((np.tile(self.qu_scaling, self.N+1), np.ones(self.N*self.n_u)))
        self.D_scaling_inv = 1/self.D_scaling

        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))
        self.du_pred = np.zeros((self.N, self.n_u))
        self.u_prev = np.zeros(self.n_u)

        self.u_ws = np.zeros((self.N+1, self.n_u))
        self.du_ws = np.zeros((self.N, self.n_u))

        self.initialized = False

        self._build_solver()

        self.state_input_prediction = None

    def initialize(self):
        pass
    
    def set_warm_start(self, u_ws: np.ndarray, du_ws: np.ndarray, state: VehicleState = None, params = None, l_ws: np.ndarray = None):
        self.u_ws = u_ws
        self.u_prev = u_ws[0]
        self.du_ws = du_ws

        if l_ws is not None:
            self.l_ws = l_ws

        if not self.initialized and state:
            q0, _ = self.dynamics.state2qu(state)
            q_ws = self._evaluate_dynamics(q0, self.u_ws[1:])

            D = np.concatenate((np.hstack((q_ws, self.u_ws)).ravel(), self.du_ws.ravel()))
            P = np.concatenate((q0, self.u_prev))
            if params is not None:
                P = np.concatenate((P, params))
            damping = 0.5

            for _ in range(2):
                # Evaluate QP approximation
                if self.qp_interface == 'casadi':
                    D_bar, success, status = self._solve_casadi(D, P)
                elif self.qp_interface == 'cvxpy':
                    D_bar, success, status = self._solve_cvxpy(D, P)
                if not success:
                    self.print_method('QP returned ' + str(status))
                    break
                D = damping*D + (1-damping)*D_bar
                D[(self.N+1)*self.n_z:] = 0

            qu_sol = D[:self.n_z*(self.N+1)].reshape((self.N+1, self.n_z))
            du_sol = D[self.n_z*(self.N+1):self.n_z*(self.N+1)+self.n_u*self.N].reshape((self.N, self.n_u))
            
            self.u_ws = qu_sol[:,self.n_q:self.n_z]
            self.du_ws = du_sol

            if self.delay_buffer is not None:
                for i in range(self.dynamics.n_u):
                    self.delay_buffer.append(deque(self.u_ws[1:1+self.delay[i],i], maxlen=self.delay[i]))

            self.initalized = True

    def step(self, vehicle_state, params=None):
        self.solve(vehicle_state, params)

        # u = self.u_pred[self.delay]
        u = self.u_pred[0]
        self.dynamics.qu2state(vehicle_state, None, u)
        if self.state_input_prediction is None:
            self.state_input_prediction = VehiclePrediction()
        self.dynamics.qu2prediction(self.state_input_prediction, self.q_pred, self.u_pred)
        self.state_input_prediction.t = vehicle_state.t

        # Update delay buffer
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer[i].append(u[i])

        # Construct initial guess for next iteration
        u_ws = np.vstack((self.u_pred, self.u_pred[-1]))
        du_ws = np.vstack((self.du_pred[1:], self.du_pred[-1]))
        self.set_warm_start(u_ws, du_ws)

        return

    def solve(self, state: VehicleState, params: np.ndarray = None):
        q0, _ = self.dynamics.state2qu(state)

        if self.delay_buffer is not None:
            delay_steps = int(np.amin(self.delay))
            u_delay = np.hstack([np.array(self.delay_buffer[i])[:delay_steps].reshape((-1,1)) for i in range(self.dynamics.n_u)])
            q_bar = self._evaluate_dynamics(q0, u_delay)
            q0 = q_bar[-1]
        q_ws = self._evaluate_dynamics(q0, self.u_ws[1:])
        if self.wrapped_state_idxs is not None:
            for i, p in zip(self.wrapped_state_idxs, self.wrapped_state_periods):
                q_ws[:,i] = np.unwrap(q_ws[:,i], period=p)

        D = np.concatenate((np.hstack((q_ws, self.u_ws)).ravel(), 
                            self.du_ws.ravel()))
        P = np.concatenate((q0, self.u_prev))
        if params is not None:
            P = np.concatenate((P, params))

        for _ in range(self.qp_iters):
            if self.qp_interface == 'casadi':
                D_bar, success, status = self._solve_casadi(D, P)
            elif self.qp_interface == 'cvxpy':
                D_bar, success, status = self._solve_cvxpy(D, P)
            if not success:
                self.print_method('Warning: QP returned ' + str(status))
                break
            D = self.damping*D + (1-self.damping)*D_bar
            D[self.n_z*(self.N+1):] = 0

        if success:
            # Unpack solution
            qu_sol = D[:self.n_z*(self.N+1)].reshape((self.N+1, self.n_z))
            du_sol = D[self.n_z*(self.N+1):self.n_z*(self.N+1)+self.n_u*self.N].reshape((self.N, self.n_u))

            u_sol = qu_sol[1:,self.n_q:self.n_q+self.n_u]
            q_sol = qu_sol[:,:self.n_q]
        else:
            q_sol = q_ws
            u_sol = self.u_ws[1:]
            du_sol = self.du_ws

        self.q_pred = q_sol
        self.u_pred = u_sol
        self.du_pred = du_sol

    def get_prediction(self):
        return self.state_input_prediction

    def _evaluate_dynamics(self, q0, U):
        t = time.time()
        Q = [q0]
        for k in range(U.shape[0]):
            Q.append(self.dynamics.fd(Q[k], U[k]).toarray().squeeze())
        if self.verbose:
            self.print_method('Dynamics evalution time: ' + str(time.time()-t))
        return np.array(Q)

    def _build_solver(self):
        # Dynamcis augmented with arc length dynamics
        q_sym = ca.MX.sym('q', self.n_q)
        u_sym = ca.MX.sym('u', self.n_u)

        # Exact disretization of the linearized continuous time dynamics with zero-order hold
        # Ac = ca.jacobian(self.dynamics.fc(q_sym, u_sym), q_sym)
        # Bc = ca.jacobian(self.dynamics.fc(q_sym, u_sym), u_sym)
        # gc = self.dynamics.fc(q_sym, u_sym) - Ac @ q_sym - Bc @ u_sym

        # H = self.dt*ca.vertcat(ca.horzcat(Ac, Bc, gc), ca.DM.zeros(self.n_u+1, self.n_q+self.n_u+1))
        # M = ca.expm(H)
        # Ad = M[:self.n_q,:self.n_q]
        # Bd = M[:self.n_q,self.n_q:self.n_q+self.n_u]
        # gd = M[:self.n_q,self.n_q+self.n_u]
        # self.f_Ad = ca.Function('Ad', [q_sym, u_sym], [Ad])
        # self.f_Bd = ca.Function('Bd', [q_sym, u_sym], [Bd])
        # self.f_gd = ca.Function('gd', [q_sym, u_sym], [gd])
        
        # Use default discretization scheme
        self.f_Ad = self.dynamics.fAd
        self.f_Bd = self.dynamics.fBd
        self.f_gd = ca.Function('gd', [q_sym, u_sym], [self.dynamics.fd(q_sym, u_sym) - self.f_Ad(q_sym, u_sym) @ q_sym - self.f_Bd(q_sym, u_sym) @ u_sym])

        # q_0, ..., q_N
        q_ph = [ca.MX.sym(f'q_ph_{k}', self.n_q) for k in range(self.N+1)] # State
        # u_-1, u_0, ..., u_N-1
        u_ph = [ca.MX.sym(f'u_ph_{k}', self.n_u) for k in range(self.N+1)] # Inputs
        # du_0, ..., du_N-1
        du_ph = [ca.MX.sym(f'du_ph_{k}', self.n_u) for k in range(self.N)] # Input rates

        qu0_ph = ca.MX.sym('qu0', self.n_z) # Initial state

        # Scaling matricies
        T_q = ca.DM(sp.sparse.diags(self.q_scaling))
        T_q_inv = ca.DM(sp.sparse.diags(self.q_scaling_inv))
        T_u = ca.DM(sp.sparse.diags(self.u_scaling))
        T_u_inv = ca.DM(sp.sparse.diags(self.u_scaling_inv))
        T_qu = ca.DM(sp.sparse.diags(self.qu_scaling))
        T_qu_inv = ca.DM(sp.sparse.diags(self.qu_scaling_inv))

        pJq_ph = []
        pJu_ph = []
        pJdu_ph = []
        pCqu_ph = []

        A, B, g = [], [], []
        Q_qu, q_qu, Q_du, q_du = [], [], [], []
        C_qu, C_qu_lb, C_qu_ub, C_du, C_du_lb, C_du_ub = [], [], [], [], [], []
        for k in range(self.N+1):
            _Q_qu = ca.MX.sym(f'_Q_qu_{k}', ca.Sparsity(self.n_z, self.n_z))
            _q_qu = ca.MX.sym(f'_q_qu_{k}', ca.Sparsity(self.n_z, 1))

            # Quadratic approximation of state costs
            if self.costs['state'][k]:
                if self.costs['state'][k].n_in() == 2:
                    pq_k = ca.MX.sym(f'pq_{k}', self.costs['state'][k].numel_in(1))
                    Jq_k = self.costs['state'][k](q_ph[k], pq_k)
                    pJq_ph.append(pq_k)
                else:
                    Jq_k = self.costs['state'][k](q_ph[k])
            else:
                Jq_k = ca.DM.zeros(1)
            M_q = ca.jacobian(ca.jacobian(Jq_k, q_ph[k]), q_ph[k])
            m_q = ca.jacobian(Jq_k, q_ph[k]).T
            _Q_qu[:self.n_q,:self.n_q] = T_q_inv @ M_q @ T_q_inv
            _q_qu[:self.n_q] = T_q_inv @ (m_q - M_q @ q_ph[k])

            # Quadratic approximation of input costs
            if self.costs['input'][k]:
                if self.costs['input'][k].n_in() == 2:
                    pu_k = ca.MX.sym(f'pu_{k}', self.costs['input'][k].numel_in(1))
                    Ju_k = self.costs['input'][k](u_ph[k], pu_k)
                    pJu_ph.append(pu_k)
                else:
                    Ju_k = self.costs['input'][k](u_ph[k])
            else:
                Ju_k = ca.DM.zeros(1)
            M_u = ca.jacobian(ca.jacobian(Ju_k, u_ph[k]), u_ph[k])
            m_u = ca.jacobian(Ju_k, u_ph[k]).T
            _Q_qu[self.n_q:,self.n_q:] = T_u_inv @ M_u @ T_u_inv
            _q_qu[self.n_q:] = T_u_inv @ (m_u - M_u @ u_ph[k])

            Q_qu.append((_Q_qu + _Q_qu.T)/2 + 1e-10*ca.DM.eye(self.n_q+self.n_u))
            q_qu.append(_q_qu)

            _C_qu, _C_qu_ub, _C_qu_lb = ca.MX.sym(f'_C_qu_{k}', 0, self.n_z), ca.MX.sym(f'_C_qu_ub_{k}', 0), ca.MX.sym(f'_C_qu_lb_{k}', 0)
            # Linear approximation of constraints on states and inputs
            if self.constraints['state_input'][k]:
                if self.constraints['state_input'][k].n_in() == 3:
                    pqu_k = ca.MX.sym(f'pqu_{k}', self.constraints['state_input'][k].numel_in(2))
                    C = self.constraints['state_input'][k](q_ph[k], u_ph[k], pqu_k)
                    pCqu_ph.append(pqu_k)
                else:
                    C = self.constraints['state_input'][k](q_ph[k], u_ph[k])
                _C = ca.jacobian(C, ca.vertcat(q_ph[k], u_ph[k]))
                _C_ub = -C + _C @ ca.vertcat(q_ph[k], u_ph[k])
                _C_lb = -1e10*np.ones(_C_ub.size1())
                
                _C_qu = ca.vertcat(_C_qu, _C)
                _C_qu_ub = ca.vertcat(_C_qu_ub, _C_ub)
                _C_qu_lb = ca.vertcat(_C_qu_lb, _C_lb)

            self.n_c[k] += _C_qu.size1()
            C_qu.append(_C_qu @ T_qu_inv)
            C_qu_ub.append(_C_qu_ub)
            C_qu_lb.append(_C_qu_lb)

            if k < self.N:
                # Linearized dynamics
                _A = ca.MX.sym(f'_A_{k}', ca.Sparsity(self.n_z, self.n_z))
                _A[:self.n_q,:self.n_q] = T_q @ self.f_Ad(q_ph[k], u_ph[k]) @ T_q_inv
                _A[:self.n_q,self.n_q:] = T_q @ self.f_Bd(q_ph[k], u_ph[k]) @ T_u_inv
                _A[self.n_q:,self.n_q:] = ca.DM.eye(self.n_u)
                A.append(_A)

                _B = ca.MX.sym(f'_B_{k}', ca.Sparsity(self.n_z, self.n_u))
                _B[:self.n_q,:] = T_q @ self.f_Bd(q_ph[k], u_ph[k]) @ T_u_inv
                _B[self.n_q:,:] = ca.DM.eye(self.n_u)
                B.append(_B)

                _g = ca.MX.sym(f'_g_{k}', ca.Sparsity(self.n_z, 1))
                _g[:self.n_q] = T_q @ self.f_gd(q_ph[k], u_ph[k])
                _g[self.n_q:] = ca.DM.zeros(self.n_u)
                g.append(_g)

                # Quadratic approximation of input rate costs
                if self.costs['rate'][k]:
                    if self.costs['rate'][k].n_in() == 2:
                        pdu_k = ca.MX.sym(f'pdu_{k}', self.costs['rate'][k].numel_in(1))
                        Jdu_k = self.costs['rate'][k](du_ph[k], pdu_k)
                        pJdu_ph.append(pdu_k)
                    else:
                        Jdu_k = self.costs['rate'][k](du_ph[k])
                else:
                    Jdu_k = ca.DM.zeros(1)
                M_du = ca.jacobian(ca.jacobian(Jdu_k, du_ph[k]), du_ph[k])
                m_du = ca.jacobian(Jdu_k, du_ph[k]).T
                Q_du.append(M_du)
                q_du.append(m_du - M_du @ du_ph[k])

                # Linear approximation of constraints on input rates
                _C_du, _C_du_ub, _C_du_lb = ca.MX.sym(f'_C_du_{k}', 0, self.n_u), ca.MX.sym(f'_C_du_ub_{k}', 0), ca.MX.sym(f'_C_du_lb_{k}', 0)
                if self.constraints['rate'][k]:
                    _C_du = ca.jacobian(self.constraints['rate'][k](du_ph[k]), du_ph[k])
                    _C_du_ub = -self.constraints['rate'][k](du_ph[k]) + _C_du @ du_ph[k]
                    _C_du_lb = -1e10*np.ones(_C_du.size1())
                
                self.n_c[k] += _C_du.size1()
                C_du.append(_C_du)
                C_du_ub.append(_C_du_ub)
                C_du_lb.append(_C_du_lb)
        
        # Form decision vector using augmented states (q_k, u_k) and inputs du_k
        # D = [(q_0, u_-1), ..., (q_N, u_N-1), du_0, ..., du_N-1]
        D = []
        for q, u in zip(q_ph, u_ph):
            D.extend([q, u])
        D += du_ph
        if self.soft_state_bound_idxs is not None and self.qp_interface in ['casadi']:
            state_ub_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))
            state_lb_slack_ph = ca.MX.sym('state_ub_slack_ph', len(self.soft_state_bound_idxs))
            D += [state_ub_slack_ph, state_lb_slack_ph]
        
        # Parameters
        P = [qu0_ph] + pJq_ph + pJu_ph + pJdu_ph + pCqu_ph

        D = ca.vertcat(*D)
        P = ca.vertcat(*P)

        n_D = D.size1()

        if self.qp_interface == 'casadi':
            # Construct batch QP cost matrix and vector
            H = ca.MX.sym('H', ca.Sparsity(n_D, n_D))
            h = ca.MX.sym('h', ca.Sparsity(n_D, 1))
            for k in range(self.N+1):
                H[k*self.n_z:(k+1)*self.n_z,k*self.n_z:(k+1)*self.n_z]  = Q_qu[k]
                h[k*self.n_z:(k+1)*self.n_z]                            = q_qu[k]
                if k < self.N:
                    s_idx, e_idx = (self.N+1)*self.n_z+k*self.n_u, (self.N+1)*self.n_z+(k+1)*self.n_u
                    H[s_idx:e_idx,s_idx:e_idx]  = Q_du[k]
                    h[s_idx:e_idx]              = q_du[k]
            
            # Soft state bounds slack cost
            if self.soft_state_bound_idxs is not None:
                n_s = 2*len(self.soft_state_bound_idxs)
                s_idx = (self.N+1)*self.n_z + self.N*self.n_u 
                H[s_idx:s_idx+n_s,s_idx:s_idx+n_s] = 2*ca.DM(sp.sparse.diags(np.concatenate((self.soft_state_bound_quad, self.soft_state_bound_quad))))
                h[s_idx:s_idx+n_s] = np.concatenate((self.soft_state_bound_lin, self.soft_state_bound_lin))

            # H = (H + H.T)/2
            self.f_H = ca.Function('H', [D, P], [H])
            self.f_h = ca.Function('h', [D, P], [h])
            
            # Construct equality constraint matrix and vector
            A_eq = ca.MX.sym('A_eq', ca.Sparsity((self.N+1)*self.n_z, n_D))
            b_eq = ca.MX.sym('b_eq', ca.Sparsity((self.N+1)*self.n_z, 1))
            A_eq[:self.n_z,:self.n_z] = ca.DM.eye(self.n_z)
            b_eq[:self.n_z] = T_qu @ qu0_ph
            for k in range(self.N):
                A_eq[(k+1)*self.n_z:(k+2)*self.n_z,k*self.n_z:(k+1)*self.n_z] = -A[k]
                A_eq[(k+1)*self.n_z:(k+2)*self.n_z,(k+1)*self.n_z:(k+2)*self.n_z] = ca.DM.eye(self.n_z)
                A_eq[(k+1)*self.n_z:(k+2)*self.n_z,(self.N+1)*self.n_z+k*self.n_u:(self.N+1)*self.n_z+(k+1)*self.n_u] = -B[k]
                b_eq[(k+1)*self.n_z:(k+2)*self.n_z] = g[k]
            self.f_A_eq = ca.Function('A_eq', [D, P], [A_eq])
            self.f_b_eq = ca.Function('b_eq', [D, P], [b_eq])

            # Construct inequality constraint matrix and vectors
            n_Cqu = int(np.sum([c.size1() for c in C_qu]))
            n_Cdu = int(np.sum([c.size1() for c in C_du]))
            n_C = n_Cqu + n_Cdu
            if self.soft_state_bound_idxs is not None and self.qp_interface in ['casadi']:
                n_C += 2*self.N*len(self.soft_state_bound_idxs)

            A_in = ca.MX.sym('A_in', ca.Sparsity(n_C, n_D))
            ub_in = ca.MX.sym('ub_in', ca.Sparsity(n_C, 1))
            lb_in = ca.MX.sym('lb_in', ca.Sparsity(n_C, 1))
            s1_idx, s2_idx = 0, n_Cqu
            for k in range(self.N+1):
                n_c = C_qu[k].size1()
                A_in[s1_idx:s1_idx+n_c,k*self.n_z:(k+1)*self.n_z] = C_qu[k]
                ub_in[s1_idx:s1_idx+n_c] = C_qu_ub[k]
                lb_in[s1_idx:s1_idx+n_c] = C_qu_lb[k]
                s1_idx += n_c
                if k < self.N:
                    n_c = C_du[k].size1()
                    A_in[s2_idx:s2_idx+n_c,(self.N+1)*self.n_z+k*self.n_u:(self.N+1)*self.n_z+(k+1)*self.n_u] = C_du[k]
                    ub_in[s2_idx:s2_idx+n_c] = C_du_ub[k]
                    lb_in[s2_idx:s2_idx+n_c] = C_du_lb[k]
                    s2_idx += n_c

            if self.soft_state_bound_idxs is not None:
                rs_idx = n_Cqu + n_Cdu
                cs_idx = (self.N+1)*self.n_z + self.N*self.n_u
                for i, j in enumerate(self.soft_state_bound_idxs):
                    for k in range(self.N):
                        # q[j] <= ub[j] + s[j]
                        A_in[rs_idx+2*k,(k+1)*self.n_z+j] = 1 * T_q_inv[j,j]
                        A_in[rs_idx+2*k,cs_idx+i] = -1
                        ub_in[rs_idx+2*k] = self.state_ub[j]
                        lb_in[rs_idx+2*k] = -1e10
                        # q[j] >= lb[j] - s[j]
                        A_in[rs_idx+2*k+1,(k+1)*self.n_z+j] = 1 * T_q_inv[j,j]
                        A_in[rs_idx+2*k+1,cs_idx+len(self.soft_state_bound_idxs)+i] = 1
                        lb_in[rs_idx+2*k+1] = self.state_lb[j]
                        ub_in[rs_idx+2*k+1] = 1e10

            self.f_A_in = ca.Function('A_in', [D, P], [A_in])
            self.f_ub_in = ca.Function('ub_in', [D, P], [ub_in])
            self.f_lb_in = ca.Function('lb_in', [D, P], [lb_in])

        # Functions which return the QP components for each stage
        self.f_Qqu = ca.Function('Qqu', [D, P], Q_qu)
        self.f_qqu = ca.Function('qqu', [D, P], q_qu)
        self.f_Qdu = ca.Function('Qdu', [D, P], Q_du)
        self.f_qdu = ca.Function('qdu', [D, P], q_du)
        self.f_Cqu = ca.Function('Cqu', [D, P], C_qu)
        self.f_Cquub = ca.Function('Cqu_ub', [D, P], C_qu_ub)
        self.f_Cqulb = ca.Function('Cqu_lb', [D, P], C_qu_lb)
        self.f_Cdu = ca.Function('Cdu', [D, P], C_du)
        self.f_Cduub = ca.Function('Cdu_ub', [D, P], C_du_ub)
        self.f_Cdulb = ca.Function('Cdu_lb', [D, P], C_du_lb)
        self.f_A = ca.Function('A', [D, P], A)
        self.f_B = ca.Function('B', [D, P], B)
        self.f_g = ca.Function('g', [D, P], g)

        if self.qp_interface == 'casadi':
            prob = dict(h=H.sparsity(), a=ca.vertcat(A_eq.sparsity(), A_in.sparsity()))
            # solver = 'qrqp'
            # solver_opts = dict(error_on_fail=False)
            solver = 'osqp'
            solver_opts = dict(error_on_fail=False, osqp={'polish': True, 'verbose': False})
            self.solver = ca.conic('qp', solver, prob, solver_opts)
        elif self.qp_interface == 'cvxpy':
            pass
        else:
            raise(ValueError('QP interface name not recognized'))
    
    def _solve_casadi(self, D, P,):
        if self.verbose:
            self.print_method('============ Sovling using CaSAdi ============')
        t = time.time()
        D_scaling = copy.copy(self.D_scaling)
        state_ub = copy.copy(self.state_ub)
        state_lb = copy.copy(self.state_lb)
        if self.soft_state_bound_idxs is not None:
            state_ub[self.soft_state_bound_idxs] = np.inf
            state_lb[self.soft_state_bound_idxs] = -np.inf
            D = np.concatenate((D, np.zeros(2*len(self.soft_state_bound_idxs))))
            D_scaling = np.concatenate((D_scaling, np.ones(2*len(self.soft_state_bound_idxs))))
        D_scaling_inv = 1/D_scaling

        qu_ub = np.concatenate((state_ub, self.input_ub))
        qu_lb = np.concatenate((state_lb, self.input_lb))
        D_lb = np.concatenate((-np.inf*np.ones(self.n_q), self.input_lb, np.tile(qu_lb, self.N), np.tile(self.input_rate_lb, self.N)))
        D_ub = np.concatenate((np.inf*np.ones(self.n_q), self.input_ub, np.tile(qu_ub, self.N), np.tile(self.input_rate_ub, self.N)))
        if self.soft_state_bound_idxs is not None:
            D_lb = np.concatenate((D_lb, np.zeros(2*len(self.soft_state_bound_idxs))))
            D_ub = np.concatenate((D_ub, np.inf*np.ones(2*len(self.soft_state_bound_idxs))))
        lb = D_scaling * D_lb
        ub = D_scaling * D_ub

        # Evaluate QP approximation
        h       = self.f_h(D, P)
        H       = self.f_H(D, P)
        A_eq    = self.f_A_eq(D, P)
        b_eq    = self.f_b_eq(D, P)
        A_in    = self.f_A_in(D, P)
        ub_in   = self.f_ub_in(D, P)
        lb_in   = self.f_lb_in(D, P)
        if self.verbose:
            self.print_method('Evaluation time: ' + str(time.time()-t))

        t = time.time()
        sol = self.solver(h=H, g=h, a=ca.vertcat(A_eq, A_in), lba=ca.vertcat(b_eq, lb_in), uba=ca.vertcat(b_eq, ub_in), lbx=lb, ubx=ub)
        if self.verbose:
            self.print_method('Solve time: ' + str(time.time()-t))

        t = time.time()
        D_bar = None
        success = self.solver.stats()['success']
        status = self.solver.stats()['return_status']

        if success:
            t = time.time()
            D_bar = (D_scaling_inv*sol['x'].toarray().squeeze())[:(self.N+1)*self.n_z+self.N*self.n_u]
        if self.verbose:
            self.print_method('Unpack time: ' + str(time.time()-t))
            self.print_method('==============================================')

        return D_bar, success, status

    def _solve_cvxpy(self, D, P):
        if self.verbose:
            self.print_method('============ Sovling using CVXPY ============')
        t = time.time()
        Q = self.f_Qqu(D, P)
        q = self.f_qqu(D, P)
        R = self.f_Qdu(D, P)
        r = self.f_qdu(D, P)
        Cqu = self.f_Cqu(D, P)
        lbCqu = self.f_Cqulb(D, P)
        ubCqu = self.f_Cquub(D, P)
        Cdu = self.f_Cdu(D, P)
        lbCdu = self.f_Cdulb(D, P)
        ubCdu = self.f_Cduub(D, P)
        A = self.f_A(D, P)
        B = self.f_B(D, P)
        g = self.f_g(D, P)

        if self.verbose:
            self.print_method('Evaluation time: ' + str(time.time()-t))

        t = time.time()
        x = cp.Variable(shape=(self.n_z, self.N+1))
        u = cp.Variable(shape=(self.n_u, self.N))
        if self.soft_state_bound_idxs is not None:
            soft_idxs = self.soft_state_bound_idxs
            state_ub_slack = cp.Variable(shape=(len(soft_idxs), 1))
            state_lb_slack = cp.Variable(shape=(len(soft_idxs), 1))
        
        if self.soft_state_bound_idxs is not None:
            hard_idxs = np.setdiff1d(np.arange(self.n_z), soft_idxs)
            qu_scaling = self.qu_scaling[hard_idxs]
            lb = qu_scaling * np.concatenate((self.state_lb, self.input_lb))[hard_idxs]
            ub = qu_scaling * np.concatenate((self.state_ub, self.input_ub))[hard_idxs]
            lb_soft = self.q_scaling[soft_idxs] * self.state_lb[soft_idxs]
            ub_soft = self.q_scaling[soft_idxs] * self.state_ub[soft_idxs]
        else:
            lb = self.qu_scaling * np.concatenate((self.state_lb, self.input_lb))
            ub = self.qu_scaling * np.concatenate((self.state_ub, self.input_ub))
        qu0 = P[:self.n_z]

        constraints, objective = [x[:,0] == self.qu_scaling*qu0], 0
        for k in range(self.N+1):
            objective += 0.5*cp.quad_form(x[:,k], Q[k].full()) + q[k].full().T @ x[:,k]
            if k < self.N:
                objective += 0.5*cp.quad_form(u[:,k], R[k].full()) + r[k].full().T @ u[:,k]
            if k > 0:
                if self.soft_state_bound_idxs is not None:
                    constraints += [x[soft_idxs,k] <= ub_soft + state_ub_slack, x[soft_idxs,k] >= lb_soft - state_lb_slack]
                    if len(hard_idxs) > 0:
                        constraints += [x[hard_idxs,k] <= ub, x[hard_idxs,k] >= lb]
                else:
                    constraints += [x[:,k] <= ub, x[:,k] >= lb]
                if Cqu[k].size1() > 0:
                    constraints += [Cqu[k].full() @ x[:,k] <= ubCqu[k].full().squeeze(), Cqu[k].full() @ x[:,k] >= lbCqu[k].full().squeeze()]
            if k < self.N:
                constraints += [x[:,k+1] == A[k].full() @ x[:,k] + B[k].full() @ u[:,k] + g[k].full().squeeze()]
                constraints += [u[:,k] <= self.input_rate_ub, u[:,k] >= self.input_rate_lb]
                if Cdu[k].size1() > 0:
                    constraints += [Cdu[k].full() @ x[:,k] <= ubCdu[k].full().squeeze(), Cdu[k].full() @ x[:,k] >= lbCdu[k].full().squeeze()]
        
        if self.soft_state_bound_idxs is not None:
            constraints += [state_ub_slack >= 0, state_lb_slack >= 0]
            objective += 0.5*cp.quad_form(state_ub_slack, np.diag(self.soft_state_bound_quad)) + self.soft_state_bound_lin @ state_ub_slack
            objective += 0.5*cp.quad_form(state_lb_slack, np.diag(self.soft_state_bound_quad)) + self.soft_state_bound_lin @ state_lb_slack

        prob = cp.Problem(cp.Minimize(objective), constraints)
        if self.verbose:
            self.print_method('Construction time: ' + str(time.time()-t))
        
        t = time.time()
        success = False
        D_bar = None
        prob.solve(solver='ECOS', verbose=self.verbose)
        if self.verbose:
            self.print_method('Solve time: ' + str(time.time()-t))

        t = time.time()
        if prob.status == 'optimal':
            success = True
            qu_sol, du_sol = x.value.T @ np.diag(self.qu_scaling_inv), u.value.T
            D_bar = np.concatenate((qu_sol.ravel(), du_sol.ravel()))
        if self.verbose:
            self.print_method('Unpack time: ' + str(time.time()-t))
            self.print_method('==============================================')

        return D_bar, success, prob.status