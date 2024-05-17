#!/usr/bin python3

import numpy as np
import scipy as sp
import casadi as ca

import os
import pathlib
import copy
import shutil
import pdb
import time
import itertools

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
jl.using("PyCall")
jl.using("PATHSolver")

import matplotlib.pyplot as plt

from typing import List, Dict

from DGSQP.dynamics.dynamics_models import CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.types import VehicleState, VehiclePrediction

from DGSQP.solvers.abstract_solver import AbstractSolver
from DGSQP.solvers.solver_types import PATHMCPParams

class PATHMCP(AbstractSolver):
    def __init__(self, joint_dynamics: CasadiDecoupledMultiAgentDynamicsModel, 
                       costs: List[List[ca.Function]], 
                       agent_constraints: List[ca.Function], 
                       shared_constraints: List[ca.Function],
                       bounds: Dict[str, VehicleState],
                       params=PATHMCPParams(),
                       print_method=print,
                       xy_plot=None,
                       use_mx=False,
                       pose_idx=[0, 1]):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a
        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method

        self.N                  = params.N

        self.q_c            = 0.1
        self.q_l            = 1000

        self.f_cl = []
        for a in range(self.M):
            _f_cl = self.joint_dynamics.dynamics_models[a].get_contouring_lag_costs_quad_approx(self.q_c, self.q_l)
            self.f_cl.append(_f_cl)

        self.f_tb = []
        for a in range(self.M):
            _f_tb = self.joint_dynamics.dynamics_models[a].get_track_boundary_constraint_lin_approx()
            self.f_tb.append(_f_tb)

        # Convergence tolerance
        self.p_tol              = params.p_tol

        self.nms                = params.nms
        self.nms_mstep_frequency = params.nms_frequency
        self.nms_memory_size    = params.nms_memory_size

        self.verbose            = params.verbose
        self.save_iter_data     = params.save_iter_data

        self.solver_name        = params.solver_name

        if use_mx:
            self.ca_sym = ca.MX.sym
        else:
            self.ca_sym = ca.SX.sym

        self.debug_plot         = params.debug_plot
        self.pause_on_plot      = params.pause_on_plot

        # The costs should be a dict of casadi functions with keys 'stage' and 'terminal'
        if len(costs) != self.M:
            raise ValueError('Number of agents: %i, but %i cost functions were provided' % (self.M, len(costs)))
        self.costs_sym = costs

        # The constraints should be a list (of length N+1) of casadi functions such that constraints[i] <= 0
        self.constraints_sym = agent_constraints
        self.shared_constraints_sym = shared_constraints

        # Process box constraints
        self.state_ub, self.state_lb, self.input_ub, self.input_lb = [], [], [], []
        self.state_ub_idxs, self.state_lb_idxs, self.input_ub_idxs, self.input_lb_idxs = [], [], [], []
        for a in range(self.M):
            su, iu = self.joint_dynamics.dynamics_models[a].state2qu(bounds['ub'][a])
            sl, il = self.joint_dynamics.dynamics_models[a].state2qu(bounds['lb'][a])
            self.state_ub.append(su)
            self.state_lb.append(sl)
            self.input_ub.append(iu)
            self.input_lb.append(il)
            self.state_ub_idxs.append(np.where(su < np.inf)[0])
            self.state_lb_idxs.append(np.where(sl > -np.inf)[0])
            self.input_ub_idxs.append(np.where(iu < np.inf)[0])
            self.input_lb_idxs.append(np.where(il > -np.inf)[0])

        self.n_cs = [0 for _ in range(self.N+1)]
        self.n_ca = [[0 for _ in range(self.N+1)] for _ in range(self.M)]
        self.n_cbr = [[0 for _ in range(self.N+1)] for _ in range(self.M)]
        self.n_c = [0 for _ in range(self.N+1)]
        
        self.state_input_predictions = [VehiclePrediction() for _ in range(self.M)]

        self.n_u = self.joint_dynamics.n_u
        self.n_q = self.joint_dynamics.n_q
            
        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))

        self.q_new = np.zeros((self.N+1, self.n_q))
        self.u_new = np.zeros((self.N+1, self.n_u))

        self.num_qa_d = [int(self.joint_dynamics.dynamics_models[a].n_q) for a in range(self.M)]
        self.num_ua_d = [int(self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]
        self.num_ua_el = [int(self.N*self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]

        self.ua_idxs = [np.concatenate([np.arange(int(self.n_u*k+np.sum(self.num_ua_d[:a])), int(self.n_u*k+np.sum(self.num_ua_d[:a+1]))) for k in range(self.N)]) for a in range(self.M)]

        self._build_solver()
        
        self.u_prev = np.zeros(self.n_u)
        self.l_pred = np.zeros(np.sum(self.n_c))
        self.u_ws = np.zeros((self.N, self.n_u)).ravel()
        self.l_ws = None

        self.initialized = True

        if self.debug_plot:
            self.colors = ['b', 'g', 'r', 'm', 'c']
            plt.ion()
            self.fig = plt.figure(figsize=(20,10))
            
            self.ax_xy = self.fig.add_subplot(1, 1+self.M, 1)
            self.ax_q = [[self.fig.add_subplot(self.num_qa_d[i]+self.num_ua_d[i], 1+self.M, i+2 + j*(1+self.M)) for j in range(self.num_qa_d[i])] for i in range(self.M)]
            self.ax_u = [[self.fig.add_subplot(self.num_qa_d[i]+self.num_ua_d[i], 1+self.M, i+2 + (j+self.num_qa_d[i])*(1+self.M)) for j in range(self.num_ua_d[i])] for i in range(self.M)]
            
            self.l_xy = []
            self.l_q = [[] for _ in range(self.M)]
            self.l_u = [[] for _ in range(self.M)]

            if xy_plot is not None:
                xy_plot(self.ax_xy)
            self.pose_idx = pose_idx

            for i in range(self.M):
                self.l_xy.append(self.ax_xy.plot([], [], f'{self.colors[i]}o')[0])
                self.ax_q[i][0].set_title(f'Agent {i+1}')
                for j in range(self.num_qa_d[i]):
                    self.l_q[i].append(self.ax_q[i][j].plot([], [], f'{self.colors[i]}o')[0])
                    self.ax_q[i][j].set_ylabel(f'State {j+1}')
                    self.ax_q[i][j].get_xaxis().set_ticks([])
                for j in range(self.num_ua_d[i]):
                    self.l_u[i].append(self.ax_u[i][j].plot([], [], f'{self.colors[i]}s')[0])
                    self.ax_u[i][j].set_ylabel(f'Input {j+1}')
                    self.ax_u[i][j].get_xaxis().set_ticks([])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _solve_mcp(self, u, l, x0, up, 
                    reference: List[np.ndarray] = [],
                    parameters: np.ndarray = np.array([])):
        Main.z0 = np.concatenate((u, l))

        Main.ub = np.inf*np.ones(self.n_u*self.N + np.sum(self.n_c))
        Main.lb = np.concatenate((-np.inf*np.ones(self.n_u*self.N), np.zeros(np.sum(self.n_c))))

        Main.nnz = self.J.numel_out(0)

        def _F_py(z):
            # Evaluate MPCC specific cost and constraints
            _P, _ = self._evaluate_mpcc(z[:self.n_u*self.N], x0, reference)
            P = np.concatenate((parameters, _P))
            _F = np.array(self.F(z, x0, up, P)).squeeze()
            return _F

        def _J_py(z):
            # Evaluate MPCC specific cost and constraints
            _P, _ = self._evaluate_mpcc(z[:self.n_u*self.N], x0, reference)
            P = np.concatenate((parameters, _P))
            _J = np.array(self.J(z, x0, up, P))
            return _J

        Main.F_py = _F_py
        Main.J_py = _J_py

        Main.tol = self.p_tol

        F_def = """
        function F(n::Cint, x::Vector{Cdouble}, f::Vector{Cdouble})
            @assert n == length(x)
            f .= F_py(x)
            return Cint(0)
        end
        return(F)
        """
        Main.F = jl.eval(F_def)

        J_def = """
        function J(
            n::Cint,
            nnz::Cint,
            x::Vector{Cdouble},
            col::Vector{Cint},
            len::Vector{Cint},
            row::Vector{Cint},
            data::Vector{Cdouble},
        )
            @assert n == length(x)  == length(col) == length(len)
            @assert nnz == length(row) == length(data)
            j = Array{Float64}(undef, n, n)
            j .= J_py(x)
            i = 1
            for c in 1:n
                col[c], len[c] = i, 0
                for r in 1:n
                    # if !iszero(j[r, c])
                    #     row[i], data[i] = r, j[r, c]
                    #     len[c] += 1
                    #     i += 1
                    # end
                    row[i], data[i] = r, j[r, c]
                    len[c] += 1
                    i += 1
                end
            end
            return Cint(0)
        end
        return(J)
        """
        Main.J = jl.eval(J_def)
    
        if self.verbose:
            output = 'yes'
        else:
            output = 'no'
        
        if self.nms:
            nms = 'yes'
        else:
            nms = 'no'

        solve = f"""
        PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")
        status, z, info = PATHSolver.solve_mcp(F, 
                                               J,
                                               lb,
                                               ub,
                                               z0,
                                               nnz=nnz,
                                               output="{output}",
                                               convergence_tolerance=tol,
                                               nms="{nms}",
                                               crash_nbchange_limit=1000,
                                               major_iteration_limit=100000,
                                               minor_iteration_limit=100000,
                                               cumulative_iteration_limit=100000,
                                               restart_limit=100)
        success = status == PATHSolver.MCP_Solved

        return z, success, info.residual, status
        """
        z, success, res, status = jl.eval(solve)

        _P, _ = self._evaluate_mpcc(z[:self.n_u*self.N], x0, reference)
        P = np.concatenate((parameters, _P))

        # _f = np.array(self.F(z, x0, up, parameters)).squeeze()
        _f = np.array(self.F(z, x0, up, P)).squeeze()
        _g = -_f[self.n_u*self.N:]
        _u, _l = z[:self.n_u*self.N], z[self.n_u*self.N:]

        stat = np.linalg.norm(_f[:self.n_u*self.N], ord=np.inf)
        feas = max(0, np.amax(_g))
        comp = np.linalg.norm(_g * _l, ord=np.inf)

        return _u, _l, success, feas, comp, stat, status.__name__

    def initialize(self):
        pass

    def set_warm_start(self, u_ws: np.ndarray, l_ws: np.ndarray = None):
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.n_u:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (u_ws.shape[0],u_ws.shape[1],self.N,self.n_u)))
        
        u = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_d[:a]))
            ei = int(np.sum(self.num_ua_d[:a])+self.num_ua_d[a])
            u.append(u_ws[:,si:ei].ravel())
        self.u_ws = np.concatenate(u)
        self.l_ws = l_ws

    def step(self, states: List[VehicleState],
             reference: List[np.ndarray] = [],
             parameters: np.ndarray = np.array([])):
        info = self.solve(states, reference, parameters)

        self.joint_dynamics.qu2state(states, None, self.u_pred[0])
        self.joint_dynamics.qu2prediction(self.state_input_predictions, self.q_pred, self.u_pred)
        for q in self.state_input_predictions:
            q.t = states[0].t

        self.u_prev = self.u_pred[0]

        if info['msg'] not in ['diverged', 'qp_fail']:
            u_ws = np.vstack((self.u_pred[1:], self.u_pred[-1]))
            self.set_warm_start(u_ws)

        return info

    def get_prediction(self) -> List[VehiclePrediction]:
        return self.state_input_predictions

    def solve(self, states: List[VehicleState], 
              reference: List[np.ndarray] = [],
              parameters: np.ndarray = np.array([])):
        
        u = copy.copy(self.u_ws)
        up = copy.copy(self.u_prev)

        x0 = self.joint_dynamics.state2q(states)

        if len(reference) == 0:
            reference = [np.zeros(self.N+1) for _ in range(self.M)]

        # Evaluate MPCC specific cost and constraints
        _P, _ = self._evaluate_mpcc(u, x0, reference)
        P = np.concatenate((parameters, _P))

        if self.debug_plot:
            self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up), copy.copy(P))
            if self.pause_on_plot:
                pdb.set_trace()
                
        solve_start = time.time()
        # Warm start dual variables
        q, G, _, _ = self._evaluate(u, None, x0, up, P=P, hessian=False)
        G = G.sparse()
        l = np.maximum(0, -sp.sparse.linalg.lsqr(G @ G.T, G @ q)[0])
        if l is None:
            l = np.zeros(np.sum(self.n_c))
        init = dict(u=u, l=l)

        # self.print_method(self.solver_name)
        u, l, converged, feas, comp, stat, msg = self._solve_mcp(u, l, x0, up, reference, parameters)
        self.print_method(f'p feas: {feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e}')
        cond = {'p_feas': feas, 'comp': comp, 'stat': stat}

        x_bar = np.array(self.evaluate_dynamics(u, x0)).squeeze()
        ua_bar = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_el[:a]))
            ei = int(np.sum(self.num_ua_el[:a])+self.num_ua_el[a])
            ua_bar.append(u[si:ei].reshape((self.N, self.num_ua_d[a])))
        u_bar = np.hstack(ua_bar)

        self.q_pred = x_bar
        self.u_pred = u_bar
        self.l_pred = l

        # Evaluate MPCC specific cost and constraints
        _P, _ = self._evaluate_mpcc(u, x0, reference)
        P = np.concatenate((parameters, _P))
        J = np.array(self.f_J(u, x0, up, P)).squeeze()

        solve_dur = time.time() - solve_start
        self.print_method(f'Solve status: {msg}')
        # self.print_method(f'Solve iters: {iters}')
        self.print_method(f'Solve time: {solve_dur:.2f}')
        self.print_method(str(J))

        solve_info = {}
        solve_info['time'] = solve_dur
        # solve_info['num_iters'] = iters
        solve_info['status'] = converged
        solve_info['cost'] = J
        solve_info['conds'] = cond
        # solve_info['iter_data'] = iter_data
        solve_info['msg'] = msg
        solve_info['init'] = init
        solve_info['primal_sol'] = u
        solve_info['dual_sol'] = l
        solve_info['x_pred'] = x_bar
        solve_info['u_pred'] = u_bar

        if self.debug_plot:
                self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up), copy.copy(P))
                if self.pause_on_plot:
                    pdb.set_trace()

        if self.debug_plot:
            plt.ioff()

        return solve_info

    def _evaluate(self, u, l, x0, up, P: np.ndarray = np.array([]), hessian: bool = True):
        eval_start = time.time()
        x = ca.vertcat(*self.evaluate_dynamics(u, x0))
        A = self.evaluate_jacobian_A(x, u)
        B = self.evaluate_jacobian_B(x, u)
        if self.N == 1:
            A = [A]
            B = [B]
        Du_x = self.f_Du_x(*A, *B)

        g = ca.vertcat(*self.f_Cxu(x, u, up, P)).full().squeeze()
        H = self.f_Du_C(x, u, up, Du_x, P)
        q = self.f_q(x, u, up, Du_x, P).full().squeeze()

        if hessian:
            E = self.evaluate_hessian_E(x, u)
            F = self.evaluate_hessian_F(x, u)
            G = self.evaluate_hessian_G(x, u)
            Q = self.f_Q(x, u, l, up, *A, *B, *E, *F, *G, P)
            # eval_time = time.time() - eval_start
            # if self.verbose:
            #     self.print_method(f'Jacobian and Hessian evaluation time: {eval_time}')
            return Q, q, H, g, x
        else:
            # eval_time = time.time() - eval_start
            # if self.verbose:
            #     self.print_method(f'Jacobian evaluation time: {eval_time}')
            return q, H, g, x
        
    def _evaluate_mpcc(self, u, x0, z):
        eval_start = time.time()
        x = self.evaluate_dynamics(u, x0)

        P = []
        _s = 0
        for a in range(self.M):
            G, g, Q, q = [], [], [], []
            for k in range(self.N+1):
                _G, _g = self.f_tb[a](x[k][_s:_s+self.num_qa_d[a]])
                _Q, _q = self.f_cl[a](x[k][_s:_s+self.num_qa_d[a]], z[a][k])
                if (not _Q.is_regular()) or (not _q.is_regular()) or (not _G.is_regular()) or (not _g.is_regular()):
                    pdb.set_trace()
                Q.append(ca.vec(_Q))
                q.append(_q)
                G.append(ca.vec(_G))
                g.append(_g)
            _s += self.num_qa_d[a]
            P += Q + q + G + g
        P = np.array(ca.vertcat(*P)).squeeze()

        eval_time = time.time() - eval_start
        if self.verbose:
            self.print_method(f'Lag and contouring approximation evaluation time: {eval_time}')

        return P, ca.vertcat(*x)

    def _build_solver(self):
        # u_0, ..., u_N-1, u_-1
        u_ph = [[self.ca_sym(f'u_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N+1)] for a in range(self.M)] # Agent inputs
        ua_ph = [ca.vertcat(*u_ph[a][:-1]) for a in range(self.M)] # [u_0^1, ..., u_{N-1}^1], [u_0^2, ..., u_{N-1}^2], ...
        uk_ph = [ca.vertcat(*[u_ph[a][k] for a in range(self.M)]) for k in range(self.N+1)] # [[u_0^1, u_0^2], ..., [u_{N-1}^1, u_{N-1}^2]]

        agent_cost_params = [[] for _ in range(self.M)]
        agent_constraint_params = [[] for _ in range(self.M)]
        shared_constraint_params = []

        # Function for evaluating the dynamics function given an input sequence
        xr_ph = [self.ca_sym('xr_ph_0', self.n_q)] # Initial state
        for k in range(self.N):
            xr_ph.append(self.joint_dynamics.fd(xr_ph[k], uk_ph[k]))
        self.evaluate_dynamics = ca.Function('evaluate_dynamics', [ca.vertcat(*ua_ph), xr_ph[0]], xr_ph)

        # State sequence placeholders
        x_ph = [self.ca_sym(f'x_ph_{k}', self.n_q) for k in range(self.N+1)]

        # Function for evaluating the dynamics Jacobians given a state and input sequence
        A, B = [], []
        for k in range(self.N):
            A.append(self.joint_dynamics.fAd(x_ph[k], uk_ph[k]))
            B.append(self.joint_dynamics.fBd(x_ph[k], uk_ph[k]))
        self.evaluate_jacobian_A = ca.Function('evaluate_jacobian_A', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], A)
        self.evaluate_jacobian_B = ca.Function('evaluate_jacobian_B', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], B)

        # Placeholders for dynamics Jacobians
        # [Dx0_x1, Dx1_x2, ..., DxN-1_xN]
        A_ph = [self.ca_sym(f'A_ph_{k}', self.joint_dynamics.sym_Ad.sparsity()) for k in range(self.N)]
        # [Du0_x1, Du1_x2, ..., DuN-1_xN]
        B_ph = [self.ca_sym(f'B_ph_{k}', self.joint_dynamics.sym_Bd.sparsity()) for k in range(self.N)]

        # Function for evaluating the dynamics Hessians given a state and input sequence
        E, F, G = [], [], []
        for k in range(self.N):
            E += self.joint_dynamics.fEd(x_ph[k], uk_ph[k])
            F += self.joint_dynamics.fFd(x_ph[k], uk_ph[k])
            G += self.joint_dynamics.fGd(x_ph[k], uk_ph[k])
        self.evaluate_hessian_E = ca.Function('evaluate_hessian_E', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], E)
        self.evaluate_hessian_F = ca.Function('evaluate_hessian_F', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], F)
        self.evaluate_hessian_G = ca.Function('evaluate_hessian_G', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], G)

        # Placeholders for dynamics Hessians
        E_ph, F_ph, G_ph = [], [], []
        for k in range(self.N):
            Ek, Fk, Gk = [], [], []
            for i in range(self.n_q):
                Ek.append(self.ca_sym(f'E{k}_ph_{i}', self.joint_dynamics.sym_Ed[i].sparsity()))
                Fk.append(self.ca_sym(f'F{k}_ph_{i}', self.joint_dynamics.sym_Fd[i].sparsity()))
                Gk.append(self.ca_sym(f'G{k}_ph_{i}', self.joint_dynamics.sym_Gd[i].sparsity()))
            E_ph.append(Ek)
            F_ph.append(Fk)
            G_ph.append(Gk)

        Du_x = []
        for k in range(self.N):
            Duk_x = [self.ca_sym(f'Du{k}_x', ca.Sparsity(self.n_q*(k+1), self.n_u)), B_ph[k]]
            for t in range(k+1, self.N):
                Duk_x.append(A_ph[t] @ Duk_x[-1])
            Du_x.append(ca.vertcat(*Duk_x))
        Du_x = ca.horzcat(*Du_x)
        Du_x = ca.horzcat(*[Du_x[:,self.ua_idxs[a]] for a in range(self.M)])
        self.f_Du_x = ca.Function('f_Du_x', A_ph + B_ph, [Du_x])

        Du_x_ph = self.ca_sym('Du_x', Du_x.sparsity())

        # Agent cost functions
        # J = [ca.DM.zeros(1) for _ in range(self.M)]
        J = [[] for _ in range(self.M)]
        for a in range(self.M):
            for k in range(self.N):
                if self.costs_sym[a][k].n_in == 4:
                    pJa_k = self.ca_sym(f'pJ{a}_{k}', self.costs_sym[a][k].numel_in(3))
                    J[a].append(self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1], pJa_k))
                    agent_cost_params[a].append(pJa_k)
                else:
                    J[a].append(self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1]))
            if self.costs_sym[a][-1].n_in == 2:
                pJa_k = self.ca_sym(f'pJ{a}_{self.N}', self.costs_sym[a][k].numel_in(1))
                J[a].append(self.costs_sym[a][-1](x_ph[-1], pJa_k))
                agent_cost_params[a].append(pJa_k)
            else:
                J[a].append(self.costs_sym[a][-1](x_ph[-1]))
        
        _J = copy.deepcopy(J)

        # MPCC specific costs
        agent_Q_cl = [[] for _ in range(self.M)]
        agent_q_cl = [[] for _ in range(self.M)]
        _s = 0
        for a in range(self.M):
            for k in range(self.N+1):
                _Q = self.ca_sym(f'Qcl{a}_{k}', self.f_cl[a].sparsity_out(0))
                _q = self.ca_sym(f'qcl{a}_{k}', self.num_qa_d[a])
                x = x_ph[k][_s:_s+self.num_qa_d[a]]
                J[a][k] += (1/2)*ca.bilin(_Q, x, x) + ca.dot(_q, x)
                agent_Q_cl[a].append(ca.vec(_Q))
                agent_q_cl[a].append(_q)
            _s += self.num_qa_d[a]

        # First derivatives of cost function w.r.t. input sequence        
        Dx_Jxu = [ca.jacobian(ca.sum1(ca.vertcat(*J[a])), ca.vertcat(*x_ph)) for a in range(self.M)]
        Du_Jxu = [ca.jacobian(ca.sum1(ca.vertcat(*J[a])), ca.vertcat(*ua_ph)) for a in range(self.M)]
        Du_J = [(Du_Jxu[a] + Dx_Jxu[a] @ Du_x_ph).T for a in range(self.M)]
        Du_J = [[Du_J[a][int(np.sum(self.num_ua_el[:b])):int(np.sum(self.num_ua_el[:b+1]))] for b in range(self.M)] for a in range(self.M)]

        # Second derivatves of cost function w.r.t. input sequence using dynamic programming
        Duu_J = []
        for a in range(self.M):
            Duu_Q = []
            Dxu_Q = []
            Dx_Q = [ca.jacobian(J[a][-1], x_ph[-1])]
            Dxx_Q = [ca.jacobian(ca.jacobian(J[a][-1], x_ph[-1]), x_ph[-1])]
            for k in range(self.N-1, -1, -1):
                if k == self.N-1:
                    Jk = J[a][k]
                else:
                    Jk = J[a][k] + J[a][k+1]
                Dx_Jk = ca.jacobian(Jk, x_ph[k])
                Dxx_Jk = ca.jacobian(ca.jacobian(Jk, x_ph[k]), x_ph[k])
                Duu_Jk = ca.jacobian(ca.jacobian(Jk, uk_ph[k]), uk_ph[k])
                Dxu_Jk = ca.jacobian(ca.jacobian(Jk, uk_ph[k]), x_ph[k])
                Duu_Jk2 = ca.jacobian(ca.jacobian(Jk, uk_ph[k+1]), uk_ph[k])

                Dx_Qk = Dx_Jk + Dx_Q[-1] @ A_ph[k]

                A1 = Duu_Jk + B_ph[k].T @ Dxx_Q[-1] @ B_ph[k]
                for i in range(self.n_q):
                    A1 += Dx_Q[-1][i] * F_ph[k][i]
                if len(Dxu_Q) == 0:
                    Duu_Qk = A1
                else:
                    B1 = Dxu_Q[-1] @ B_ph[k]
                    B1[:Duu_Jk2.size1(),:] += Duu_Jk2
                    Duu_Qk = ca.blockcat([[A1, B1.T], [B1, Duu_Q[-1]]])

                A2 = Dxu_Jk + B_ph[k].T @ Dxx_Q[-1] @ A_ph[k]
                for i in range(self.n_q):
                    A2 += Dx_Q[-1][i] * G_ph[k][i]
                if len(Dxu_Q) == 0:
                    Dxu_Qk = A2
                else:
                    B2 = Dxu_Q[-1] @ A_ph[k]
                    Dxu_Qk = ca.vertcat(A2, B2)

                Dxx_Qk = Dxx_Jk + A_ph[k].T @ Dxx_Q[-1] @ A_ph[k]
                for i in range(self.n_q):
                    Dxx_Qk += Dx_Q[-1][i] * E_ph[k][i]
                
                Dx_Q.append(Dx_Qk)
                Dxx_Q.append(Dxx_Qk)
                Duu_Q.append(Duu_Qk)
                Dxu_Q.append(Dxu_Qk)
            Duu_Ja = ca.horzcat(*[Duu_Q[-1][:,self.ua_idxs[a]] for a in range(self.M)])
            Duu_Ja = ca.vertcat(*[Duu_Ja[self.ua_idxs[a],:] for a in range(self.M)])
            Duu_J.append(Duu_Ja)

        # Placeholders for gradient of dynamics w.r.t. state and input
        Cs = [[] for _ in range(self.N+1)] # Shared constraints
        Ca = [[[] for _ in range(self.N+1)] for _ in range(self.M)] # Agent specific constraints
        for k in range(self.N):
            # Add shared constraints
            if self.shared_constraints_sym[k] is not None:
                if self.shared_constraints_sym[k].n_in() == 4:
                    pCs_k = self.ca_sym(f'pCs_{k}', self.shared_constraints_sym[k].numel_in(3))
                    Cs[k].append(self.shared_constraints_sym[k](x_ph[k], uk_ph[k], uk_ph[k-1], pCs_k))
                    shared_constraint_params.append(pCs_k)
                else:
                    Cs[k].append(self.shared_constraints_sym[k](x_ph[k], uk_ph[k], uk_ph[k-1]))
            if len(Cs[k]) > 0:
                Cs[k] = ca.vertcat(*Cs[k])
                self.n_cs[k] = Cs[k].shape[0]
            else:
                Cs[k] = ca.DM()
            # Add agent constraints
            for a in range(self.M):
                if self.constraints_sym[a][k] is not None:
                    if self.constraints_sym[a][k].n_in() == 4:
                        pCa_k = self.ca_sym(f'pC{a}_{k}', self.constraints_sym[a][k].numel_in(3))
                        Ca[a][k].append(self.constraints_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1], pCa_k))
                        agent_constraint_params[a].append(pCa_k)
                    else:
                        Ca[a][k].append(self.constraints_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1]))
                # Add agent box constraints
                if len(self.input_ub_idxs[a]) > 0:
                    Ca[a][k].append(u_ph[a][k][self.input_ub_idxs[a]] - self.input_ub[a][self.input_ub_idxs[a]])
                if len(self.input_lb_idxs[a]) > 0:
                    Ca[a][k].append(self.input_lb[a][self.input_lb_idxs[a]] - u_ph[a][k][self.input_lb_idxs[a]])
                if k > 0:
                    if len(self.state_ub_idxs[a]) > 0:
                        Ca[a][k].append(x_ph[k][self.state_ub_idxs[a]+int(np.sum(self.num_qa_d[:a]))] - self.state_ub[a][self.state_ub_idxs[a]])
                    if len(self.state_lb_idxs[a]) > 0:
                        Ca[a][k].append(self.state_lb[a][self.state_lb_idxs[a]] - x_ph[k][self.state_lb_idxs[a]+int(np.sum(self.num_qa_d[:a]))])
                if len(Ca[a][k]) > 0:
                    Ca[a][k] = ca.vertcat(*Ca[a][k])
                    self.n_ca[a][k] = Ca[a][k].shape[0]
                else:
                    Ca[a][k] = ca.DM()
        # Add shared constraints
        if self.shared_constraints_sym[-1] is not None:
            if self.shared_constraints_sym[-1].n_in() == 2:
                pCs_k = self.ca_sym(f'pCs_{self.N}', self.shared_constraints_sym[-1].numel_in(1))
                Cs[-1].append(self.shared_constraints_sym[-1](x_ph[-1], pCs_k))
                shared_constraint_params.append(pCs_k)
            else:
                Cs[-1].append(self.shared_constraints_sym[-1](x_ph[-1]))
        if len(Cs[-1]) > 0:
            Cs[-1] = ca.vertcat(*Cs[-1])
            self.n_cs[-1] = Cs[-1].shape[0]
        else:
            Cs[-1] = ca.DM()
        # Add agent constraints
        for a in range(self.M):
            if self.constraints_sym[a][-1] is not None:
                if self.constraints_sym[a][-1].n_in() == 2:
                    pCa_k = self.ca_sym(f'pC{a}_{self.N}', self.constraints_sym[a][-1].numel_in(1))
                    Ca[a][-1].append(self.constraints_sym[a][-1](x_ph[-1], pCa_k))
                    agent_constraint_params[a].append(pCa_k)
                else:
                    Ca[a][-1].append(self.constraints_sym[a][-1](x_ph[-1]))
            # Add agent box constraints
            if len(self.state_ub_idxs[a]) > 0:
                Ca[a][-1].append(x_ph[-1][self.state_ub_idxs[a]+int(np.sum(self.num_qa_d[:a]))] - self.state_ub[a][self.state_ub_idxs[a]])
            if len(self.state_lb_idxs[a]) > 0:
                Ca[a][-1].append(self.state_lb[a][self.state_lb_idxs[a]] - x_ph[-1][self.state_lb_idxs[a]+int(np.sum(self.num_qa_d[:a]))])
            if len(Ca[a][-1]) > 0:
                Ca[a][-1] = ca.vertcat(*Ca[a][-1])
                self.n_ca[a][-1] = Ca[a][-1].shape[0]
            else:
                Ca[a][-1] = ca.DM()

        # MPCC specific constraints
        agent_G_tb = [[] for _ in range(self.M)]
        agent_g_tb = [[] for _ in range(self.M)]
        _s = 0
        for a in range(self.M):
            for k in range(self.N+1):
                _G = self.ca_sym(f'G{a}_{k}', self.f_tb[a].sparsity_out(0))
                _g = self.ca_sym(f'g{a}_{k}', 2)
                x = x_ph[k][_s:_s+self.num_qa_d[a]]
                # Linear approximation of track boundary constraints
                Ca[a][k] = ca.vertcat(Ca[a][k], _G @ x + _g)
                agent_G_tb[a].append(ca.vec(_G))
                agent_g_tb[a].append(_g)
            _s += self.num_qa_d[a]

        # Joint constraint functions for both agents: C(x, u) <= 0
        C = [[] for _ in range(self.N+1)]
        # Constraint indexes specific to each best response problem at each time step
        self.Cbr_k_idxs = [[[] for _ in range(self.N+1)] for _ in range(self.M)] 
        # Constraint indexes specific to each best response problem in batch vector form
        self.Cbr_v_idxs = [[] for _ in range(self.M)]
        for k in range(self.N+1):
            C[k].append(Cs[k])
            n = self.n_cs[k]
            for a in range(self.M):
                self.Cbr_k_idxs[a][k].append(np.arange(self.n_cs[k]))
                C[k].append(Ca[a][k])
                self.Cbr_k_idxs[a][k].append(np.arange(self.n_ca[a][k]) + n)
                n += self.n_ca[a][k]
                self.Cbr_k_idxs[a][k] = np.concatenate(self.Cbr_k_idxs[a][k]).astype(int)
                self.Cbr_v_idxs[a].append((self.Cbr_k_idxs[a][k] + np.sum(self.n_c[:k])).astype(int))
            C[k] = ca.vertcat(*C[k])    
            self.n_c[k] = C[k].shape[0]
        for a in range(self.M): self.Cbr_v_idxs[a] = np.concatenate(self.Cbr_v_idxs[a])

        # First derivatives of constraints w.r.t. input sequence
        Dx_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*x_ph))
        Du_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*ua_ph))
        Du_C = Du_Cxu + Dx_Cxu @ Du_x_ph
        
        # Hessian of constraints using dynamic programming
        Duu_C = []
        for k in range(self.N+1):
            for j in range(self.n_c[k]):
                Dx_Cj = [ca.jacobian(C[k][j], x_ph[k])]
                Dxx_Cj = [ca.jacobian(ca.jacobian(C[k][j], x_ph[k]), x_ph[k])]
                if k == self.N:
                    Duu_Cj, Dxu_Cj = [], []
                else:
                    Duu_Cj = [ca.jacobian(ca.jacobian(C[k][j], uk_ph[k]), uk_ph[k])]
                    Dxu_Cj = [ca.jacobian(ca.jacobian(C[k][j], uk_ph[k]), x_ph[k])]
                for t in range(k-1, -1, -1):
                    Dx_Cjt = ca.jacobian(C[k][j], x_ph[t])
                    Dxx_Cjt = ca.jacobian(ca.jacobian(C[k][j], x_ph[t]), x_ph[t])
                    Duu_Cjt = ca.jacobian(ca.jacobian(C[k][j], uk_ph[t]), uk_ph[t])
                    Dxu_Cjt = ca.jacobian(ca.jacobian(C[k][j], uk_ph[t]), x_ph[t])

                    Dx = Dx_Cjt + Dx_Cj[-1] @ A_ph[t]

                    A1 = Duu_Cjt + B_ph[t].T @ Dxx_Cj[-1] @ B_ph[t]
                    for i in range(self.n_q):
                        A1 += Dx_Cj[-1][i] * F_ph[t][i]
                    if len(Dxu_Cj) == 0:
                        Duu = A1
                    else:
                        B1 = Dxu_Cj[-1] @ B_ph[t]
                        Duu = ca.blockcat([[A1, B1.T], [B1, Duu_Cj[-1]]])

                    A2 = Dxu_Cjt + B_ph[t].T @ Dxx_Cj[-1] @ A_ph[t]
                    for i in range(self.n_q):
                        A2 += Dx_Cj[-1][i] * G_ph[t][i]
                    if len(Dxu_Cj) == 0:
                        Dxu = A2
                    else:
                        B2 = Dxu_Cj[-1] @ A_ph[t]
                        Dxu = ca.vertcat(A2, B2)

                    Dxx = Dxx_Cjt + A_ph[t].T @ Dxx_Cj[-1] @ A_ph[t]
                    for i in range(self.n_q):
                        Dxx += Dx_Cj[-1][i] * E_ph[t][i]
                    
                    Dx_Cj.append(Dx)
                    Dxx_Cj.append(Dxx)
                    Duu_Cj.append(Duu)
                    Dxu_Cj.append(Dxu)
                Duu = self.ca_sym(f'Duu_C{k}_{j}', ca.Sparsity(self.n_u*self.N, self.n_u*self.N))
                Duu[:Duu_Cj[-1].size1(),:Duu_Cj[-1].size2()] = Duu_Cj[-1]
                Duu = ca.horzcat(*[Duu[:,self.ua_idxs[a]] for a in range(self.M)])
                Duu = ca.vertcat(*[Duu[self.ua_idxs[a],:] for a in range(self.M)])
                Duu_C.append(Duu)
        
        # Paramter vector
        P = []
        for a in range(self.M):
            P += agent_cost_params[a]
        for a in range(self.M):
            P += agent_constraint_params[a]
        P += shared_constraint_params
        # MPCC cost and constraint parameters
        for a in range(self.M):
            P += agent_Q_cl[a]
            P += agent_q_cl[a]
            P += agent_G_tb[a]
            P += agent_g_tb[a]
        P = ca.vertcat(*P)

        # Cost function in sparse form
        self.f_Jxu = ca.Function('f_Jxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], P], [ca.sum1(ca.vertcat(*J[a])) for a in range(self.M)])
        
        # Cost function in batch form
        Ju = self.f_Jxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1], P)
        self.f_J = ca.Function('f_J', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1], P], Ju)
        
        # First derivatives of cost function w.r.t. input sequence
        self.f_Du_J = [ca.Function(f'f_Du_J{a}', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], Du_J[a]) for a in range(self.M)]
        
        q = ca.vertcat(*[Du_J[a][a] for a in range(self.M)])
        self.f_q = ca.Function('f_q', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], [q])

        # Second derivatives of cost function w.r.t. input sequence
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph)) \
                    + [P]
        self.f_Duu_J = ca.Function('f_Duu_J', in_args, Duu_J)

        # Constraint function in sparse form
        self.f_Cxu = ca.Function('f_Cxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], P], C)

        # Constraint function in batch form
        Cu = self.f_Cxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1], P)
        self.f_C = ca.Function('f_C', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1], P], Cu)

        # First derivatives of constraints w.r.t. input sequence
        self.f_Du_C = ca.Function('f_Du_C', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], [Du_C])

        l_ph = self.ca_sym(f'l_ph', np.sum(self.n_c))
        lDuu_C = 0
        for j in range(np.sum(self.n_c)):
            lDuu_C += l_ph[j] * Duu_C[j]
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), l_ph, uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph)) \
                    + [P]
        self.f_lDuu_C = ca.Function('f_lDuu_C', in_args, [lDuu_C])

        # Hessian of the Lagrangian
        Q = ca.vertcat(*[Duu_J[a][int(np.sum(self.num_ua_el[:a])):int(np.sum(self.num_ua_el[:a+1])),:] for a in range(self.M)]) + lDuu_C
        self.f_Q = ca.Function('f_Q', in_args, [Q])

        # Symbolic Hessian of Lagrangian
        L = [Ju[a] + ca.dot(l_ph, ca.vertcat(*Cu)) for a in range(self.M)]
        Du_L = [ca.jacobian(L[a], ca.vertcat(*ua_ph)).T for a in range(self.M)]
        self.f_Du_L = ca.Function('f_Du_L', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1], P], Du_L)
        Duu_L = [ca.jacobian(Du_L[a], ca.vertcat(*ua_ph)) for a in range(self.M)]
        self.f_Duu_L = ca.Function('f_Duu_L', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1], P], Duu_L)

        _Du_L = [ca.jacobian(L[a], ua_ph[a]).T for a in range(self.M)]
        _Cu = ca.vertcat(*Cu)
        F = ca.vertcat(*_Du_L, -_Cu)
        self.F = ca.Function('F', [ca.vertcat(*ua_ph, l_ph), xr_ph[0], uk_ph[-1], P], [F])
        J = ca.jacobian(F, ca.vertcat(*ua_ph, l_ph))
        self.J = ca.Function('J', [ca.vertcat(*ua_ph, l_ph), xr_ph[0], uk_ph[-1], P], [J])

    def _update_debug_plot(self, u, x0, up, P=np.array([])):
        q_bar = np.array(self.evaluate_dynamics(u, x0)).squeeze()
        ua_bar = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_el[:a]))
            ei = int(np.sum(self.num_ua_el[:a])+self.num_ua_el[a])
            ua_bar.append(u[si:ei].reshape((self.N, self.num_ua_d[a])))
        u_bar = np.hstack(ua_bar)
        for i in range(self.M):
            self.l_xy[i].set_data(q_bar[:,self.pose_idx[0]+int(np.sum(self.num_qa_d[:i]))], q_bar[:,self.pose_idx[1]+int(np.sum(self.num_qa_d[:i]))])
        self.ax_xy.set_aspect('equal')
        self.ax_xy.relim()
        self.ax_xy.autoscale_view()
        J = self.f_J(u, x0, up, P)
        self.ax_xy.set_title(f'{self.solver_name} | {str(J)}')
        for i in range(self.M):
            for j in range(self.num_qa_d[i]):
                _q =  q_bar[:,j+int(np.sum(self.num_qa_d[:i]))]
                self.l_q[i][j].set_data(np.arange(self.N+1), _q)
                self.ax_q[i][j].relim()
                self.ax_q[i][j].autoscale_view()
            for j in range(self.num_ua_d[i]):
                _u = u_bar[:,j+int(np.sum(self.num_ua_d[:i]))]
                self.l_u[i][j].set_data(np.arange(self.N), _u)
                self.ax_u[i][j].relim()
                self.ax_u[i][j].autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()