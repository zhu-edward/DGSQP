#!/usr/bin python3

ZERO_TOL = 1e-8

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

from collections import deque

import matplotlib.pyplot as plt

from typing import List, Dict

from DGSQP.dynamics.dynamics_models import CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.types import VehicleState, VehiclePrediction

from DGSQP.solvers.abstract_solver import AbstractSolver
from DGSQP.solvers.solver_types import DGSQPV2Params

from dataclasses import dataclass, field

@dataclass
class IterationData():
    iteration: int                  = field(default=0)
    qp: dict                        = field(default_factory=lambda : {})
    primal_iterate: np.ndarray      = field(default=None)
    primal_step: np.ndarray         = field(default=None)
    dual_iterate: np.ndarray        = field(default=None)
    dual_step: np.ndarray           = field(default=None)
    slack_iterate: np.ndarray       = field(default=None)
    slack_step: np.ndarray          = field(default=None)
    merit_parameter: float          = field(default=0)
    primal_feasibility: float       = field(default=0)
    complementarity: float          = field(default=0)
    stationarity: float             = field(default=0)
    gradient_evaluations: int       = field(default=0)
    hessian_evaluations: int        = field(default=0)
    qp_solutions: int               = field(default=0)
    iteration_time: float           = field(default=0)
    step_type: str                  = field(default=None)
    step_size: float                = field(default=1.0)
    watchdog: bool                  = field(default=False)
    elastic_mode: bool              = field(default=False)

class DGSQP(AbstractSolver):
    def __init__(self, joint_dynamics: CasadiDecoupledMultiAgentDynamicsModel, 
                       costs: List[List[ca.Function]], 
                       agent_constraints: List[ca.Function], 
                       shared_constraints: List[ca.Function],
                       bounds: Dict[str, VehicleState],
                       params=DGSQPV2Params(),
                       use_mx=False,
                       print_method=print,
                       xy_plot=None):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a
        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method
        self.qp_interface       = params.qp_interface
        self.qp_solver          = params.qp_solver

        self.N                  = params.N

        self.reg_init           = params.reg
        self.reg_decay          = params.reg_decay
        self.line_search_iters  = params.line_search_iters
        self.sqp_iters          = params.sqp_iters
        self.merit_function     = params.merit_function
        self.merit_parameter    = params.merit_parameter
        self.merit_decrease_condition = params.merit_decrease_condition

        # Convergence tolerance for SQP
        self.p_tol              = params.p_tol
        self.d_tol              = params.d_tol
        self.rel_tol_req        = 10

        # Line search parameters
        self.beta               = params.beta
        self.tau                = params.tau

        self.verbose            = params.verbose
        self.save_iter_data     = params.save_iter_data
        self.save_qp_data       = params.save_qp_data
        if params.time_limit is None:
            self.time_limit = np.inf
        else:
            self.time_limit = params.time_limit

        self.code_gen           = params.code_gen
        self.jit                = params.jit
        self.opt_flag           = params.opt_flag
        self.solver_name        = params.solver_name

        if use_mx:
            self.ca_sym = ca.MX.sym
        else:
            self.ca_sym = ca.SX.sym

        self.debug_plot         = params.debug_plot
        self.pause_on_plot      = params.pause_on_plot
        self.local_pos          = params.local_pos

        self.options = {}

        # The costs should be a dict of casadi functions with keys 'stage' and 'terminal'
        if len(costs) != self.M:
            raise ValueError('Number of agents: %i, but %i cost functions were provided' % (self.M, len(costs)))
        self.costs_sym = costs

        # The constraints should be a list (of length N+1) of casadi functions such that constraints[i] <= 0
        # if len(constraints) != self.N+1:
        #     raise ValueError('Horizon length: %i, but %i constraint functions were provided' % (self.N+1, len(constraints)))
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

        if params.solver_dir:
            self._load_solver()
        else:
            self._build_solver()
        
        self.u_prev = np.zeros(self.n_u)
        self.l_pred = np.zeros(np.sum(self.n_c))
        self.u_ws = np.zeros((self.N, self.n_u)).ravel()
        self.l_ws = None

        # Construct QP solver
        if self.qp_interface == 'casadi':
            if self.qp_solver == 'osqp':
                solver_opts = dict(error_on_fail=False, osqp=dict(polish=True, verbose=self.verbose))
            elif self.qp_solver == 'qrqp':
                solver_opts = dict(error_on_fail=False)
            elif self.qp_solver == 'superscs':
                solver_opts = dict(error_on_fail=False)
            elif self.qp_solver == 'qpoases':
                solver_opts = dict(error_on_fail=False, sparse=False, printLevel='tabular' if self.verbose else 'none')
            elif self.qp_solver == 'cplex':
                # Change this to match version of libcplex<CPLEX_VERSION>.so 
                os.environ['CPLEX_VERSION'] = '2210'
                solver_opts = dict(error_on_fail=False, cplex=dict(CPXPARAM_OptimalityTarget=2, CPXPARAM_ScreenOutput=self.verbose))

            G_sparsity = self.f_Du_C.sparsity_out(0)
            Gem_sparsity = self.f_Du_Cem.sparsity_out(0)
            Q_sparsity = self.f_Q.sparsity_out(0)
            Qem_sparsity = self.f_Qem.sparsity_out(0)
            
            # Nominal QP solver
            self.solver = ca.conic('qp', self.qp_solver, {'h': Q_sparsity, 'a': G_sparsity}, solver_opts)
            # Elastic mode QP solver
            self.em_solver = ca.conic('em_qp', self.qp_solver, {'h': Qem_sparsity, 'a': Gem_sparsity}, solver_opts)

            self.dual_name = 'lam_a'
        else:
            raise(ValueError(f'Unsupported QP interface {self.qp_interface}'))

        self.initialized = True

        self.gamma = params.delta_decay
        self.sigma = params.merit_decrease

        self.nms = params.nms
        self.nms_mstep_frequency = params.nms_frequency
        self.nms_memory_size = params.nms_memory_size

        self.nms_initial_step_size_factor = 20
        self.nms_initial_reference_factor = 1
        
        self.soc = False
        self.soc_steps = 4
        self.soc_index = 0

        self.eta = 1e3
        self.rho = 1e3

        self.qp_fails = 0

        if self.debug_plot:
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

            for i in range(self.M):
                self.l_xy.append(self.ax_xy.plot([], [], 'o')[0])
                self.ax_q[i][0].set_title(f'Agent {i+1}')
                for j in range(self.num_qa_d[i]):
                    self.l_q[i].append(self.ax_q[i][j].plot([], [], 'o')[0])
                    self.ax_q[i][j].set_ylabel(f'State {j+1}')
                    self.ax_q[i][j].get_xaxis().set_ticks([])
                for j in range(self.num_ua_d[i]):
                    self.l_u[i].append(self.ax_u[i][j].plot([], [], 's')[0])
                    self.ax_u[i][j].set_ylabel(f'Input {j+1}')
                    self.ax_u[i][j].get_xaxis().set_ticks([])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _solve_qp(self, Q, q, G, g, l, x0=None):
        t = time.time()
        Q = self._nearest_pd(Q)

        if self.reg > 0:
            Q += self.reg*np.eye(Q.shape[0])
        if x0 is None:
            x0 = np.zeros(self.N*self.n_u)

        if self.verbose:
            self.print_method(f'Cost matrix condition number: {np.linalg.cond(Q)}')
            self.print_method(f'Constraint matrix condition number: {np.linalg.cond(G)}')

        try:
            if self.qp_interface == 'casadi':
                sol = self.solver(h=Q, g=q, a=G, uba=-g, x0=x0)
                status = self.solver.stats()['return_status']
                if self.verbose: self.print_method(f'QP status: {status}')
                if self.solver.stats()['success']:
                    du = sol['x'].toarray().squeeze()
                    dl = sol['lam_a'].toarray().squeeze() - l
                else:
                    du = [None]
                    dl = [None]
            else:
                raise(ValueError(f'Unsupported QP interface {self.qp_interface}'))
            if self.verbose: self.print_method(f'QP solve time: {(time.time()-t):.3f}')
        except Exception as e:
            self.print_method(f'QP solver failed with exception: {e}')
            du = [None]
            dl = [None]
        
        return du, dl
    
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
                parameters: np.ndarray = np.array([])):
        info = self.solve(states, parameters)

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
                parameters: np.ndarray = np.array([])):
        solve_info = {}
        solve_start = time.time()

        u = copy.copy(self.u_ws)
        up = copy.copy(self.u_prev)

        x0 = self.joint_dynamics.state2q(states)

        # Warm start dual variables
        q, G, g, _ = self._evaluate(u, None, x0, up, P=parameters, hessian=False)
        G = G.sparse()
        l = np.maximum(0, -sp.sparse.linalg.lsqr(G @ G.T, G @ q)[0])
        if l is None:
            l = np.zeros(np.sum(self.n_c))
        init = dict(u=u, l=l)
        u_im1 = copy.copy(u)
        l_im1 = copy.copy(l)
        
        phi = float(self.f_phi(u, l, np.maximum(0, g), q, G, g, 1.0, x0, up, parameters))
        self.nms_memory = deque([self.nms_initial_reference_factor*phi], self.nms_memory_size)
        self.reg = copy.copy(self.reg_init)

        self.checkpoint_counter = 0
        self.checkpoint_index = 0
        self.checkpoint_delta = 0
        self.checkpoint_reg = self.reg

        if self.debug_plot:
            self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up), copy.copy(parameters))
            if self.pause_on_plot:
                pdb.set_trace()

        sqp_converged = False
        finished = False
        rel_tol_its = 0
        sqp_it = 0
        m_step_it = 0
        iter_data = []
        while True:
            if self.verbose:
                self.print_method('===================================================')
                self.print_method(f'DGSQP iteration: {sqp_it}')
            sqp_it_start = time.time()

            _data = IterationData(iteration=sqp_it,
                                  primal_iterate=copy.copy(u),
                                  dual_iterate=copy.copy(l))

            qp_solves = 0

            # Evaluate SQP approximation
            Q_i, q_i, G_i, g_i, _ = self._evaluate(u, l, x0, up, P=parameters)

            d_i = q_i + G_i.T @ l

            xtol = self.p_tol
            ltol = self.d_tol
            _data.primal_feasibility = max(0, np.amax(g_i))
            _data.complementarity = np.linalg.norm(g_i * l, ord=np.inf)
            _data.stationarity = np.linalg.norm(d_i, ord=np.inf)
            if self.verbose:
                self.print_method(f'p feas: {_data.primal_feasibility:.4e} | comp: {_data.complementarity:.4e} | stat: {_data.stationarity:.4e}')
            if self.debug_plot:
                if len(iter_data) > 0:
                    if iter_data[-1].step_type == 'm-step':
                        self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up), copy.copy(parameters))
                        if self.pause_on_plot:
                            pdb.set_trace()

            # Divergence
            if _data.stationarity > 1e10:
                sqp_converged = False
                finished = True
                msg = 'diverged'
                if self.verbose: self.print_method('SQP diverged')
            # Converged via optimaility conditions
            if _data.primal_feasibility < xtol and _data.complementarity < ltol and _data.stationarity < ltol:
                sqp_converged = True
                finished = True
                msg = 'conv_abs_tol'
                if self.verbose: self.print_method('SQP converged via optimality conditions')
            # Max iterations reached
            # if sqp_it >= self.sqp_iters:
            if m_step_it >= self.sqp_iters:
                sqp_converged = False
                finished = True
                msg = 'max_it'
                if self.verbose: self.print_method('Max SQP iterations reached')
            if time.time() - solve_start > self.time_limit:
                sqp_converged = False
                finished = True
                msg = 'time_limit_exceeded'
                if self.verbose: self.print_method('Time limit exceeded')

            if finished:
                _data.qp_solutions = qp_solves
                _data.iteration_time = time.time() - sqp_it_start
                iter_data.append(_data)
                break

            # Compute SQP primal dual step
            if self.save_qp_data:
                _data.qp = dict(Q=Q_i, q=q_i, G=G_i, g=g_i)
                _data.qp['L'] = ca.vertcat(*self.f_Duu_L(u, l, x0, up, parameters))
                _data.qp['B'] = self._nearest_pd(_data.qp['L'])
            du, dl = self._solve_qp(Q_i, q_i, G_i, g_i, l)
            qp_solves += 1

            if None in du:
                if self.nms:
                    self.qp_fails += 1
                    # if self.qp_fails >= 10:
                    #     iter_data.append(_data)
                    #     sqp_converged = False
                    #     finished = True
                    #     msg = 'qp_fail'
                    #     break

                    # If QP fails, do m-step from last checkpoint
                    if self.verbose: self.print_method('QP solution failed')

                    if len(iter_data) == 0:
                        iter_data.append(_data)
                        sqp_converged = False
                        finished = True
                        msg = 'qp_fail'
                        break

                    d_step = False
                    m_step = True

                    _idx = min(self.checkpoint_index, len(iter_data)-1)
                    if self.verbose: self.print_method(f'Performing m-step from iteration {_idx}')

                    u, du, l, dl, s, ds, mu, _data = self.load_checkpoint(iter_data, _idx, _data)
                else:
                    iter_data.append(_data)
                    sqp_converged = False
                    finished = True
                    msg = 'qp_fail'
                    break
            else:
                self.qp_fails = 0
                _data.primal_step = copy.copy(du)
                _data.dual_step = copy.copy(dl)
                if sqp_it == 0:
                    # Initialize step size limit
                    self.delta = self.nms_initial_step_size_factor*np.linalg.norm(np.concatenate((du, dl)))
                    self.checkpoint_delta = copy.copy(self.delta)

                if self.nms:
                    # Determine which type of step to take
                    d_step = False
                    m_step = False
                    if self.checkpoint_counter >= self.nms_mstep_frequency:
                        if self.verbose: self.print_method('- Checkpoint reached, checking for merit decrease')
                        m_step = True
                    else:
                        step_norm = np.linalg.norm(np.concatenate((du, dl)))
                        if self.verbose: self.print_method(f'- step norm: {step_norm:.3E} | delta: {self.delta:.3E}')
                        if step_norm < self.delta:
                            d_step = True
                        else:
                            m_step = True
                else:
                    d_step = False
                    m_step = False

                s = np.maximum(0, g_i)
                ds = np.maximum(0, g_i + np.array(G_i @ du).squeeze()) - s
                _data.slack_iterate = copy.copy(s)
                _data.slack_step = copy.copy(ds)

                if self.merit_parameter is None:
                    mu = self._get_mu(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i, x0, up, parameters)
                else:
                    mu = self.merit_parameter
                _data.merit_parameter = mu
            
            if d_step:
                # Relaxed step
                if self.verbose: self.print_method('d-step')
                _data.step_type = 'd-step'
                u += du
                l += dl
                self.delta = self.gamma*self.delta
                self.checkpoint_counter += 1

            if m_step:
                if self.verbose: self.print_method('m-step')
                _data.step_type = 'm-step'
                m_step_it += 1

                # Check descent condition for full step
                _u = u + du
                _l = l + dl
                _q, _G, _g, _ = self._evaluate(_u, _l, x0, up, P=parameters, hessian=False)
                _s = np.maximum(0, _g)
                
                # Evaluate merit function at relaxed step
                phi = float(self.f_phi(_u, _l, _s, _q, _G, _g, 1.0, x0, up, parameters))
                R = (1-self.sigma*1.0)*np.amax(self.nms_memory)
                if self.verbose: self.print_method(f'- merit: {phi:.3E} | reference: {R:.3E}')
                if phi <= R:
                    if self.verbose: self.print_method(f'- Decrease condition met')
                    u = _u
                    l = _l
                else:
                    if self.verbose: self.print_method(f'- Decrease condition not met')
                    if self.checkpoint_index <= len(iter_data) - 1:
                        if self.verbose: self.print_method(f'- Returning to checkpoint iterate {self.checkpoint_index}')
                        # Watchdog: reset to last checkpoint and enforce descent via line search
                        _data.watchdog = True
                        u, du, l, dl, s, ds, mu, _data = self.load_checkpoint(iter_data, self.checkpoint_index, _data)

                        # Reset delta to checkpoint delta
                        self.delta = copy.copy(self.checkpoint_delta)
                        # Reset regularization
                        self.reg = copy.copy(self.checkpoint_reg)

                    u, l, _a, phi = self.line_search(u, du, l, dl, s, x0, up, mu, parameters)
                    _data.step_size = copy.copy(_a)

                # Convergence via relative tolerance
                # if np.linalg.norm(u-u_im1) < xtol/2 and np.linalg.norm(l-l_im1) < ltol/2:
                if np.linalg.norm(u-u_im1) < xtol and np.linalg.norm(l-l_im1) < ltol:
                    rel_tol_its += 1
                    if rel_tol_its >= self.rel_tol_req and _data.primal_feasibility < xtol:
                        sqp_converged = True
                        finished = True
                        msg = 'conv_rel_tol'
                        if self.verbose: self.print_method('SQP converged via relative tolerance')
                else:
                    rel_tol_its = 0

                u_im1 = copy.copy(u)
                l_im1 = copy.copy(l)
                
                # Regularization decay
                self.reg = self.reg * self.reg_decay
                
                self.nms_memory.append(phi)
                self.checkpoint_counter = 0
                self.checkpoint_delta = copy.copy(self.delta)
                self.checkpoint_reg = copy.copy(self.reg)
                self.checkpoint_index = sqp_it + 1
                
                if self.verbose:
                    self.print_method(f'- New checkpoint at iteration {self.checkpoint_index}')
                    self.print_method(f'- Merit memory: {str(self.nms_memory)}')

            if (not d_step) and (not m_step):
                u, l, _a, phi = self.line_search(u, du, l, dl, s, x0, up, mu, parameters)
                _data.step_size = copy.copy(_a)
                # Convergence via relative tolerance
                if np.linalg.norm(u-u_im1) < xtol and np.linalg.norm(l-l_im1) < ltol:
                    rel_tol_its += 1
                    if rel_tol_its >= self.rel_tol_req and _data.primal_feasibility < xtol:
                        sqp_converged = True
                        finished = True
                        msg = 'conv_rel_tol'
                        if self.verbose: self.print_method('SQP converged via relative tolerance')
                else:
                    rel_tol_its = 0

                u_im1 = copy.copy(u)
                l_im1 = copy.copy(l)

                # Regularization decay
                self.reg = self.reg * self.reg_decay
                    
                self.nms_memory.append(phi)

                if self.verbose:
                    self.print_method(f'- Merit memory: {str(self.nms_memory)}')

            _data.qp_solutions = qp_solves
            _data.iteration_time = time.time() - sqp_it_start
            if self.verbose:
                self.print_method(f'SQP iteration {sqp_it} time: {_data.iteration_time:.2f}')
                self.print_method('===================================================')

            # pdb.set_trace()

            iter_data.append(_data)
            sqp_it += 1
        
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

        J = self.f_J(u, x0, up, parameters)

        solve_dur = time.time() - solve_start
        self.print_method(f'Solve status: {msg}')
        p_feas, comp, stat = iter_data[-1].primal_feasibility, iter_data[-1].complementarity, iter_data[-1].stationarity
        self.print_method(f'Solve stats: p feas: {p_feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e}')
        self.print_method(f'Solve iters: {sqp_it}')
        self.print_method(f'Solve time: {solve_dur:.2f}')
        self.print_method(f'Cost: {str(np.array(J).squeeze())}')

        solve_info['time'] = solve_dur
        solve_info['num_iters'] = sqp_it
        solve_info['status'] = sqp_converged
        solve_info['cost'] = J
        if self.save_iter_data:
            solve_info['iter_data'] = iter_data
        else:
            solve_info['iter_data'] = [iter_data[0], iter_data[-1]]
        solve_info['msg'] = msg
        solve_info['init'] = init
        solve_info['primal_sol'] = u
        solve_info['dual_sol'] = l
        solve_info['x_pred'] = x_bar
        solve_info['u_pred'] = u_bar
        solve_info['conds'] = {'p_feas': p_feas, 'comp': comp, 'stat': stat}

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
            eval_time = time.time() - eval_start
            if self.verbose:
                self.print_method(f'Jacobian and Hessian evaluation time: {eval_time}')
            return Q, q, H, g, x
        else:
            eval_time = time.time() - eval_start
            if self.verbose:
                self.print_method(f'Jacobian evaluation time: {eval_time}')
            return q, H, g, x

    def _get_mu(self, u, du, l, dl, s, ds, Q, q, G, g, x0, up, P):
        thresh = 0
        d_phi_c = float(self.f_dphi_c(u, du, l, dl, s, Q, q, G, g, 0, x0, up, P))
        constr_vio = s
        rho = 0.5
        
        if d_phi_c < 0 and np.sum(constr_vio) > thresh:
            if self.verbose:
                self.print_method('Case 1: negative directional derivative with constraint violation')
            mu = -d_phi_c / ((1-rho)*np.sum(constr_vio))  
        elif d_phi_c < 0 and np.sum(constr_vio) <= thresh:
            if self.verbose:
                self.print_method('Case 2: negative directional derivative no constraint violation')
            mu = 0
        elif d_phi_c >= 0 and np.sum(constr_vio) > thresh:
            if self.verbose:
                self.print_method('Case 3: positive directional derivative with constraint violation')
            mu = d_phi_c / ((1-rho)*np.sum(constr_vio))  
        elif d_phi_c >= 0 and np.sum(constr_vio) <= thresh:
            if self.verbose:
                self.print_method('Case 4: positive directional derivative no constraint violation')
            mu = 0
        if self.verbose: self.print_method(f'mu: {mu:.2f}')
        
        return mu
    
    def load_checkpoint(self, iter_data, idx, _data):
        checkpoint_data = iter_data[idx]
        u = checkpoint_data.primal_iterate
        du = checkpoint_data.primal_step
        l = checkpoint_data.dual_iterate
        dl = checkpoint_data.dual_step
        s = checkpoint_data.slack_iterate
        ds = checkpoint_data.slack_step
        mu = checkpoint_data.merit_parameter
        _data.primal_iterate = copy.copy(u)
        _data.primal_step = copy.copy(du)
        _data.dual_iterate = copy.copy(l)
        _data.dual_step = copy.copy(dl)
        _data.slack_iterate = copy.copy(s)
        _data.slack_step = copy.copy(ds)
        _data.merit_parameter = copy.copy(mu)
        if self.save_qp_data:
            _data.qp = copy.copy(checkpoint_data.qp)
        return u, du, l, dl, s, ds, mu, _data
    
    def line_search(self, u, du, l, dl, s, x0, up, mu, P):
        # Define decrease condition
        if self.merit_decrease_condition == 'max':
            decrease_condition = lambda alpha: (1-self.sigma*alpha)*np.amax(self.nms_memory)
        elif self.merit_decrease_condition == 'armijo':
            _Q, _q, _G, _g, _ = self._evaluate(u, l, x0, up, P=P, hessian=True)
            _phi = float(self.f_phi(u, l, s, _q, _G, _g, mu, x0, up, P))
            _dphi = float(self.f_dphi(u, du, l, dl, np.maximum(0, _g), _Q, _q, _G, _g, mu, x0, up, P))
            decrease_condition = lambda alpha: _phi + self.sigma*alpha*_dphi

        # Do backtracking line search
        if self.verbose: self.print_method(f'- Performing backtracking line search')
        _a = 1.0
        for i in range(self.line_search_iters):
            _u = u + _a*du
            _l = l + _a*dl
            _q, _G, _g, _ = self._evaluate(_u, _l, x0, up, P=P, hessian=False)
            _s = np.maximum(0, _g)

            # Evaluate merit function at reduced step
            phi = float(self.f_phi(_u, _l, _s, _q, _G, _g, mu, x0, up, P))
            R = decrease_condition(_a)
            if self.verbose:
                self.print_method(f'-- Line search step: {i} | merit : {phi:.3e} | reference: {R:.3e} | a: {_a:.3e}')
            # Check decrease condition
            if phi <= R:
                if self.verbose: self.print_method(f'- Decrease condition met')
                break
            else:
                _a *= self.tau
        phi = float(self.f_phi(_u, _l, _s, _q, _G, _g, 1.0, x0, up, P))
        return _u, _l, _a, phi

    def _build_solver(self):
        # u_0, ..., u_N-1, u_-1
        u_ph = [[self.ca_sym(f'u_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N+1)] for a in range(self.M)] # Agent inputs
        ua_ph = [ca.vertcat(*u_ph[a][:-1]) for a in range(self.M)] # [u_0^1, ..., u_{N-1}^1, u_0^2, ..., u_{N-1}^2]
        uk_ph = [ca.vertcat(*[u_ph[a][k] for a in range(self.M)]) for k in range(self.N+1)] # [[u_0^1, u_0^2], ..., [u_{N-1}^1, u_{N-1}^2]]

        agent_cost_params = [[] for _ in range(self.M)]
        agent_constraint_params = [[] for _ in range(self.M)]
        shared_constraint_params = []

        # Function for evaluating the dynamics function given an input sequence
        xr_ph = [self.ca_sym('xr_ph_0', self.n_q)] # Initial state
        for k in range(self.N):
            xr_ph.append(self.joint_dynamics.fd(xr_ph[k], uk_ph[k]))
        self.evaluate_dynamics = ca.Function('evaluate_dynamics', [ca.vertcat(*ua_ph), xr_ph[0]], xr_ph, self.options)

        # State sequence placeholders
        x_ph = [self.ca_sym(f'x_ph_{k}', self.n_q) for k in range(self.N+1)]

        # Function for evaluating the dynamics Jacobians given a state and input sequence
        A, B = [], []
        for k in range(self.N):
            A.append(self.joint_dynamics.fAd(x_ph[k], uk_ph[k]))
            B.append(self.joint_dynamics.fBd(x_ph[k], uk_ph[k]))
        self.evaluate_jacobian_A = ca.Function('evaluate_jacobian_A', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], A, self.options)
        self.evaluate_jacobian_B = ca.Function('evaluate_jacobian_B', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], B, self.options)

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
        self.f_Du_x = ca.Function('f_Du_x', A_ph + B_ph, [Du_x], self.options)

        Du_x_ph = self.ca_sym('Du_x', Du_x.sparsity())

        # Agent cost functions
        J = [[] for _ in range(self.M)]
        for a in range(self.M):
            for k in range(self.N):
                if self.costs_sym[a][k].n_in() == 4:
                    pJa_k = self.ca_sym(f'pJ{a}_{k}', self.costs_sym[a][k].numel_in(3))
                    J[a].append(self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1], pJa_k))
                    agent_cost_params[a].append(pJa_k)
                else:
                    J[a].append(self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1]))
            if self.costs_sym[a][-1].n_in() == 2:
                pJa_k = self.ca_sym(f'pJ{a}_{self.N}', self.costs_sym[a][-1].numel_in(1))
                J[a].append(self.costs_sym[a][-1](x_ph[-1], pJa_k))
                agent_cost_params[a].append(pJa_k)
            else:
                J[a].append(self.costs_sym[a][-1](x_ph[-1]))
        
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

        # Duu_J2 = [ca.jacobian(ca.jacobian(Ju[a], ca.vertcat(*ua_ph)), ca.vertcat(*ua_ph)) for a in range(self.M)]
        # self.f_Duu_J2 = ca.Function('f_Duu_J2', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], Duu_J2)

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
        # self.f_Cs = ca.Function('f_Cs', [ca.vertcat(*ua_ph), ca.vertcat(*x_ph), uk_ph[-1]], Cs)
        # self.f_Ca = [ca.Function(f'f_Ca{a}', [ca.vertcat(*ua_ph), ca.vertcat(*x_ph), uk_ph[-1]], Ca[a]) for a in range(self.M)]

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
        P = ca.vertcat(*P)

        n = self.N*self.n_u
        
        # Cost function in sparse form
        self.f_Jxu = ca.Function('f_Jxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], P], [ca.sum1(ca.vertcat(*J[a])) for a in range(self.M)])
        
        # Cost function in batch form
        Ju = self.f_Jxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1], P)
        self.f_J = ca.Function('f_J', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1], P], Ju, self.options)
        
        # First derivatives of cost function w.r.t. input sequence
        self.f_Du_J = [ca.Function(f'f_Du_J{a}', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], Du_J[a]) for a in range(self.M)]
        
        q = ca.vertcat(*[Du_J[a][a] for a in range(self.M)])
        self.f_q = ca.Function('f_q', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], [q], self.options)
        
        # Second derivatives of cost function w.r.t. input sequence
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph)) \
                    + [P]
        self.f_Duu_J = ca.Function('f_Duu_J', in_args, Duu_J)

        # Constraint function in sparse form
        self.f_Cxu = ca.Function('f_Cxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], P], C, self.options)

        # Constraint function in batch form
        Cu = self.f_Cxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1], P)
        self.f_C = ca.Function('f_C', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1], P], Cu, self.options)

        # First derivatives of constraints w.r.t. input sequence
        self.f_Du_C = ca.Function('f_Du_C', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph, P], [Du_C], self.options)

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
        # self.f_Q = ca.Function('f_Q', in_args, [Q])
        self.f_Q = ca.Function('f_Q', in_args, [(Q + Q.T)/2])

        # Symbolic Hessian of Lagrangian
        L = [Ju[a] + ca.dot(l_ph, ca.vertcat(*Cu)) for a in range(self.M)]
        Du_L = [ca.jacobian(L[a], ca.vertcat(*ua_ph)).T for a in range(self.M)]
        self.f_Du_L = ca.Function('f_Du_L', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1], P], Du_L)
        Duu_L = [ca.jacobian(Du_L[a], ca.vertcat(*ua_ph))[int(np.sum(self.num_ua_el[:a])):int(np.sum(self.num_ua_el[:a+1])),:] for a in range(self.M)]
        self.f_Duu_L = ca.Function('f_Duu_L', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1], P], Duu_L)

        # Merit function
        du_ph = [[self.ca_sym(f'du_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N)] for a in range(self.M)] # Agent inputs
        dua_ph = [ca.vertcat(*du_ph[a]) for a in range(self.M)] # Stack input sequences by agent
        dl_ph = self.ca_sym(f'dl_ph', np.sum(self.n_c))
        s_ph = self.ca_sym(f's_ph', np.sum(self.n_c))
        ds_ph = self.ca_sym(f'ds_ph', np.sum(self.n_c))
        mu_ph = self.ca_sym('mu_ph', 1)

        q_ph = self.ca_sym('q_ph', self.n_u*self.N)
        Q_ph = self.ca_sym('Q_ph', Q.sparsity())
        g_ph = self.ca_sym('g_ph', np.sum(self.n_c))
        H_ph = self.ca_sym('H_ph', Du_C.sparsity())

        # stat = ca.vertcat(q_ph + H_ph.T @ l_ph, ca.dot(l_ph, g_ph))
        # stat_norm = (1/2)*ca.sumsqr(stat)
        # dstat_norm = (q_ph + H_ph.T @ l_ph).T @ ca.horzcat(Q_ph, H_ph.T) @ ca.vertcat(*dua_ph, dl_ph) \
        #                 + ca.dot(l_ph, g_ph)*(l_ph.T @ H_ph @ ca.vertcat(*dua_ph) + ca.dot(dl_ph, g_ph))

        stat = ca.vertcat(*[ca.jacobian(L[a], ua_ph[a]).T for a in range(self.M)])
        stat_norm = (1/2)*ca.sumsqr(stat)
        dstat_norm = ca.dot(ca.jacobian(stat_norm, ca.vertcat(*ua_ph, l_ph)).T, ca.vertcat(*dua_ph, dl_ph))
        
        vio = mu_ph*ca.sum1(s_ph)
        dvio = -mu_ph*ca.sum1(s_ph)

        obj = ca.sum1(ca.vertcat(*Ju))
        dobj = ca.dot(ca.jacobian(obj, ca.vertcat(*ua_ph)).T, ca.vertcat(*dua_ph))

        # phi_args = [l_ph, s_ph, q_ph, H_ph, g_ph, mu_ph]
        # dphi_args = [ca.vertcat(*dua_ph), l_ph, dl_ph, s_ph, Q_ph, q_ph, H_ph, g_ph, mu_ph]
        phi_args = [ca.vertcat(*ua_ph), l_ph, s_ph, q_ph, H_ph, g_ph, mu_ph, xr_ph[0], uk_ph[-1], P]
        dphi_args = [ca.vertcat(*ua_ph), ca.vertcat(*dua_ph), l_ph, dl_ph, s_ph, Q_ph, q_ph, H_ph, g_ph, mu_ph, xr_ph[0], uk_ph[-1], P]
        if self.merit_function == 'stat_l1':
            self.f_phi = ca.Function('f_phi', phi_args, [stat_norm + vio], self.options)
            self.f_dphi = ca.Function('f_dphi', dphi_args, [dstat_norm + dvio], self.options)
            self.f_dphi_c = ca.Function('f_dphi_c', dphi_args, [dstat_norm], self.options)
        elif self.merit_function == 'sum_obj_l1':
            self.f_phi = ca.Function('f_phi', phi_args, [obj + vio], self.options)
            self.f_dphi = ca.Function('f_dphi', dphi_args, [dobj + dvio], self.options)
            self.f_dphi_c = ca.Function('f_dphi_c', dphi_args, [dobj], self.options)
        else:
            raise(ValueError(f'Merit function option {self.merit_function} not recognized'))
        self.f_dstat_norm = ca.Function('f_dstat_norm', dphi_args, [dstat_norm], self.options)

        # Elastic mode
        _q = self.ca_sym('_q', self.n_u*self.N)
        eta_ph = self.ca_sym('eta', 1)
        qem = ca.vertcat(_q, eta_ph*ca.DM.ones(np.sum(self.n_c)))
        self.f_qem = ca.Function('f_q', [_q, eta_ph], [qem], self.options)

        _Q = self.ca_sym('_Q', self.f_Q.sparsity_out(0))
        rho_ph = self.ca_sym('rho', 1)
        Qem = self.ca_sym('Qem', ca.Sparsity(n+np.sum(self.n_c), n+np.sum(self.n_c)))
        Qem[:n,:n] = _Q
        Qem[n:,n:] = rho_ph*ca.DM.eye(np.sum(self.n_c))
        self.f_Qem = ca.Function('f_Qem', [_Q, rho_ph], [Qem])
        
        _g = self.ca_sym('_g', np.sum(self.n_c))
        gem = ca.vertcat(_g, ca.DM.zeros(np.sum(self.n_c)))
        self.f_gem = ca.Function('f_gem', [_g], [gem], self.options)

        _G = self.ca_sym('_G', self.f_Du_C.sparsity_out(0))
        Du_Cem = self.ca_sym('Du_Cem', ca.Sparsity(2*np.sum(self.n_c), n+np.sum(self.n_c)))
        Du_Cem[:np.sum(self.n_c),:n] = _G
        Du_Cem[:np.sum(self.n_c),n:] = -ca.DM.eye(np.sum(self.n_c))
        Du_Cem[np.sum(self.n_c):,n:] = -ca.DM.eye(np.sum(self.n_c))
        self.f_Du_Cem = ca.Function('f_Du_Cem', [_G], [Du_Cem], self.options)

        # Casadi C code generation
        if self.code_gen and not self.jit:
            generator = ca.CodeGenerator(self.c_file_name)
            generator.add(self.evaluate_dynamics)
            generator.add(self.evaluate_jacobian_A)
            generator.add(self.evaluate_jacobian_B)
            generator.add(self.evaluate_hessian_E)
            generator.add(self.evaluate_hessian_F)
            generator.add(self.evaluate_hessian_G)
            generator.add(self.f_Du_x)

            generator.add(self.f_J)

            generator.add(self.f_Cxu)
            generator.add(self.f_Du_C)
            
            generator.add(self.f_q)
            generator.add(self.f_Q)

            generator.add(self.f_phi)
            generator.add(self.f_dphi)
            generator.add(self.f_dstat_norm)

            # Set up paths
            cur_dir = pathlib.Path.cwd()
            gen_path = cur_dir.joinpath(self.solver_name)
            c_path = gen_path.joinpath(self.c_file_name)
            if gen_path.exists():
                shutil.rmtree(gen_path)
            gen_path.mkdir(parents=True)

            os.chdir(gen_path)
            if self.verbose:
                self.print_method(f'- Generating C code for solver {self.solver_name} at {str(gen_path)}')
            generator.generate()
            # Compile into shared object
            so_path = gen_path.joinpath(self.so_file_name)
            command = f'gcc -fPIC -shared -{self.opt_flag} {c_path} -o {so_path}'
            if self.verbose:
                self.print_method(f'- Compiling shared object {so_path} from {c_path}')
                self.print_method(f'- Executing "{command}"')
            # pdb.set_trace()
            os.system(command)
            # pdb.set_trace()
            # Swtich back to working directory
            os.chdir(cur_dir)
            install_dir = self.install()

            # Load solver
            self._load_solver(str(install_dir.joinpath(self.so_file_name)))

    def _load_solver(self, solver_path=None):
        if solver_path is None:
            solver_path = str(pathlib.Path(self.solver_dir, self.so_file_name).expanduser())
        if self.verbose:
            self.print_method(f'- Loading solver from {solver_path}')
        self.evaluate_dynamics = ca.external('evaluate_dynamics', solver_path)
        self.evaluate_jacobian_A = ca.external('evaluate_jacobian_A', solver_path)
        self.evaluate_jacobian_B = ca.external('evaluate_jacobian_B', solver_path)
        self.evaluate_hessian_E = ca.external('evaluate_hessian_E', solver_path)
        self.evaluate_hessian_F = ca.external('evaluate_hessian_F', solver_path)
        self.evaluate_hessian_G = ca.external('evaluate_hessian_G', solver_path)
        self.f_Du_x = ca.external('f_Du_x', solver_path)

        self.f_J = ca.external('f_J', solver_path)
        
        self.f_Cxu = ca.external('f_Cxu', solver_path)
        self.f_Du_C = ca.external('f_Du_C', solver_path)

        self.f_q = ca.external('f_q', solver_path)
        self.f_Q = ca.external('f_Q', solver_path)
        
        self.f_phi = ca.external('f_phi', solver_path)
        self.f_dphi = ca.external('f_dphi', solver_path)
        self.f_dstat_norm = ca.external('f_dstat_norm', solver_path)

    def _nearest_pd(self, A):
        B = (A + A.T)/2
        s, U = np.linalg.eigh(B)
        # s[np.where(s < 0)[0]] = 0
        s[np.where(s < 0)[0]] = 1e-9
        C = U @ np.diag(s) @ U.T
        return (C + C.T)/2

    def _update_debug_plot(self, u, x0, up, P=np.array([])):
        q_bar = np.array(self.evaluate_dynamics(u, x0)).squeeze()
        ua_bar = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_el[:a]))
            ei = int(np.sum(self.num_ua_el[:a])+self.num_ua_el[a])
            ua_bar.append(u[si:ei].reshape((self.N, self.num_ua_d[a])))
        u_bar = np.hstack(ua_bar)
        if not self.local_pos:
            for i in range(self.M):
                self.l_xy[i].set_data(q_bar[:,0+int(np.sum(self.num_qa_d[:i]))], q_bar[:,1+int(np.sum(self.num_qa_d[:i]))])
        else:
            for i in range(self.M):
                if i < self.M-1:
                    idx = int(np.sum(self.num_qa_d[i+1:]))
                else:
                    idx = 0
                s = q_bar[:,-2-idx]
                ey = q_bar[:,-1-idx]
                x, y = [], []
                for _s, _ey in zip(s, ey):
                    _x, _y, _ = self.joint_dynamics.dynamics_models[0].track.local_to_global((_s, _ey, 0))
                    x.append(_x)
                    y.append(_y)
                self.l_xy[i].set_data(x, y)
        self.ax_xy.set_aspect('equal')
        self.ax_xy.relim()
        self.ax_xy.autoscale_view()
        J = self.f_J(u, x0, up, P)
        self.ax_xy.set_title(f'{self.solver_name} | {str(J)}  | {self.reg:.3f}')
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