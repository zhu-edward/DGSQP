#!/usr/bin python3

import numpy as np
import scipy as sp
import casadi as ca
import osqp

import pathlib
import os
import copy
import shutil
import pdb
from datetime import datetime
import itertools

import matplotlib
import matplotlib.pyplot as plt

from typing import List, Dict

from DGSQP.dynamics.dynamics_models import CasadiDecoupledMultiAgentDynamicsModel
from DGSQP.types import VehicleState, VehiclePrediction

from DGSQP.solvers.abstract_solver import AbstractSolver
from DGSQP.solvers.solver_types import DGSQPParams

class DGSQP(AbstractSolver):
    def __init__(self, joint_dynamics: CasadiDecoupledMultiAgentDynamicsModel, 
                       costs: List[List[ca.Function]], 
                       agent_constraints: List[ca.Function], 
                       shared_constraints: List[ca.Function],
                       bounds: Dict[str, VehicleState],
                       params=DGSQPParams()):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a

        self.N = params.N

        self.reg = params.reg
        self.line_search_iters = params.line_search_iters
        self.nonmono_ls = params.nonmono_ls
        self.sqp_iters = params.sqp_iters
        self.conv_approx = params.conv_approx
        self.merit_function = params.merit_function

        self.verbose = params.verbose
        self.code_gen = params.code_gen
        self.jit = params.jit
        self.opt_flag = params.opt_flag
        self.solver_name = params.solver_name
        if params.solver_dir is not None:
            self.solver_dir = os.path.join(params.solver_dir, self.solver_name)

        if not params.enable_jacobians:
            jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        else:
            jac_opts = dict()

        if self.code_gen:
            if self.jit:
                self.options = dict(jit=True, jit_name=self.solver_name, compiler='shell', jit_options=dict(compiler='gcc', flags=['-%s' % self.opt_flag], verbose=self.verbose), **jac_opts)
            else:
                self.options = dict(jit=False, **jac_opts)
                self.c_file_name = self.solver_name + '.c'
                self.so_file_name = self.solver_name + '.so'
                if params.solver_dir is not None:
                    self.solver_dir = pathlib.Path(params.solver_dir).expanduser().joinpath(self.solver_name)
        else:
            self.options = dict(jit=False, **jac_opts)

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

        # Convergence tolerance for SQP
        self.p_tol = params.p_tol
        self.d_tol = params.d_tol
        self.rel_tol_req = 5

        # Line search parameters
        self.beta = params.beta
        self.tau = params.tau

        self.debug_plot = params.debug_plot
        self.pause_on_plot = params.pause_on_plot
        self.local_pos = params.local_pos
        if self.debug_plot:
            matplotlib.use('TkAgg')
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            # self.joint_dynamics.dynamics_models[0].track.remove_phase_out()
            self.joint_dynamics.dynamics_models[0].track.plot_map(self.ax_xy, close_loop=False)
            self.l1_xy, self.l2_xy = self.ax_xy.plot([], [], 'bo', [], [], 'go')
            self.ax_a = self.fig.add_subplot(2,2,2)
            self.l1_a, self.l2_a = self.ax_a.plot([], [], '-bo', [], [], '-go')
            self.ax_a.set_ylabel('accel')
            self.ax_s = self.fig.add_subplot(2,2,4)
            self.l1_s, self.l2_s = self.ax_s.plot([], [], '-bo', [], [], '-go')
            self.ax_s.set_ylabel('steering')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))

        self.q_new = np.zeros((self.N+1, self.n_q))
        self.u_new = np.zeros((self.N+1, self.n_u))

        self.num_qa_d = [int(self.joint_dynamics.dynamics_models[a].n_q) for a in range(self.M)]
        self.num_ua_d = [int(self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]
        self.num_ua_el = [int(self.N*self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]

        self.ua_idxs = [np.concatenate([np.arange(int(self.n_u*k+np.sum(self.num_ua_d[:a])), int(self.n_u*k+np.sum(self.num_ua_d[:a+1]))) for k in range(self.N)]) for a in range(self.M)]

        self.debug = False

        self.u_prev = np.zeros(self.n_u)

        if params.solver_dir:
            self._load_solver()
        else:
            self._build_solver()
        
        self.l_pred = np.zeros(np.sum(self.n_c))
        self.u_ws = np.zeros((self.N, self.n_u))
        self.l_ws = None

        self.initialized = True

        if not self.conv_approx:
            # Build nlp solver for Newton step
            ipopt_opts = dict(max_iter=500,
                            linear_solver='ma27',
                            warm_start_init_point='yes',
                            mu_strategy='adaptive',
                            mu_init=1e-5,
                            mu_min=1e-15,
                            barrier_tol_factor=1,
                            print_level=0)
            self.solver_options = dict(error_on_fail=False, 
                                        verbose_init=self.verbose, 
                                        ipopt=ipopt_opts)
            p_sym = ca.SX.sym('p', self.N*self.n_u)
            Q_sym = ca.SX.sym('Q', self.N*self.n_u, self.N*self.n_u)
            q_sym = ca.SX.sym('q', self.N*self.n_u)
            G_sym = ca.SX.sym('G', np.sum(self.n_c), self.N*self.n_u)
            g_sym = ca.SX.sym('G', np.sum(self.n_c))
            param = ca.vertcat(ca.vertcat(*ca.horzsplit(Q_sym)), 
                                q_sym, 
                                ca.vertcat(*ca.horzsplit(G_sym)), 
                                g_sym)

            self.solver_args = {}
            self.solver_args['x0'] = np.zeros(self.N*self.n_u)
            self.solver_args['lbx'] = -np.inf*np.ones(self.N*self.n_u)
            self.solver_args['ubx'] = np.inf*np.ones(self.N*self.n_u)
            self.solver_args['lbg'] = -np.inf*np.ones(np.sum(self.n_c))
            self.solver_args['ubg'] = np.zeros(np.sum(self.n_c))

            f = ca.bilin(Q_sym, p_sym, p_sym)/2 + ca.dot(q_sym, p_sym)
            g = g_sym + G_sym @ p_sym
            nlp = dict(x=p_sym, f=f, g=g, p=param)
            self.solver = ca.nlpsol('solver', 'ipopt', nlp, self.solver_options)

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

    def step(self, states: List[VehicleState], env_state=None):
        info = self.solve(states)

        self.joint_dynamics.qu2state(states, None, self.u_pred[0])
        self.joint_dynamics.qu2prediction(self.state_input_predictions, self.q_pred, self.u_pred)
        for q in self.state_input_predictions:
            q.t = states[0].t

        self.u_prev = self.u_pred[0]

        u_ws = np.vstack((self.u_pred[1:], self.u_pred[-1]))
        self.set_warm_start(u_ws)

        return info

    def get_prediction(self) -> List[VehiclePrediction]:
        return self.state_input_predictions

    def solve(self, states: List[VehicleState]):
        solve_info = {}
        solve_start = datetime.now()
        self.u_prev = np.zeros(self.n_u)

        u = copy.copy(self.u_ws)
        up = copy.copy(self.u_prev)

        x0 = self.joint_dynamics.state2q(states)

        # Warm start dual variables
        # if self.l_ws is None:
        #     # Least squares approx
        #     q, G, _, _ = self._evaluate(u, None, x0, up, hessian=False)
        #     G = sp.sparse.csc_matrix(G)
        #     l = np.maximum(0, -sp.sparse.linalg.lsqr(G @ G.T, G @ q)[0])
        # else:
        #     l = copy.copy(self.l_ws)
        q, G, _, _ = self._evaluate(u, None, x0, up, hessian=False)
        G = sp.sparse.csc_matrix(G)
        l = np.maximum(0, -sp.sparse.linalg.lsqr(G @ G.T, G @ q)[0])
        if l is None:
            l = np.zeros(np.sum(self.n_c))
        init = dict(u=u, l=l)

        if self.debug_plot:
            self._update_debug_plot(u, x0, up)
            if self.pause_on_plot:
                pdb.set_trace()

        sqp_converged = False
        rel_tol_its = 0
        sqp_it = 0
        iter_data = []
        rho = np.zeros(self.N+1)
        print(self.solver_name)
        while True:
            sqp_it_start = datetime.now()
            if self.verbose:
                print('===================================================')
                print(f'SQGAMES iteration: {sqp_it}')

            u_im1 = copy.copy(u)
            l_im1 = copy.copy(l)

            # Evaluate SQP approximation
            Q_i, q_i, G_i, g_i, _ = self._evaluate(u, l, x0, up)
            d_i = q_i + G_i.T @ l

            # Convergence test
            xtol = self.p_tol
            ltol = self.d_tol
            p_feas = max(0, np.amax(g_i))
            comp = np.linalg.norm(g_i * l, ord=np.inf)
            stat = np.linalg.norm(d_i, ord=np.inf)
            cond = {'p_feas': p_feas, 'comp': comp, 'stat': stat}
            if self.verbose:
                print(f'SQP iteration {sqp_it}')
                print(f'p feas: {p_feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e}')
            if stat > 1e5:
                sqp_it_dur = (datetime.now()-sqp_it_start).total_seconds()
                iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                if self.verbose: print('SQP diverged')
                msg = 'diverged'
                sqp_converged = False
                break
            if p_feas < xtol and comp < ltol and stat < ltol:
                sqp_it_dur = (datetime.now()-sqp_it_start).total_seconds()
                iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                sqp_converged = True
                msg = 'conv_abs_tol'
                if self.verbose: print('SQP converged via optimality conditions')
                break
            
            # Compute SQP primal dual step
            if self.conv_approx:
                du, l_hat = self._solve_conv_qp(Q_i, q_i, G_i, g_i)
            else:
                du, l_hat = self._solve_nonconv_qp(Q_i, q_i, G_i, g_i)
            qp_solves = 1
            if None in l_hat:
                sqp_it_dur = (datetime.now()-sqp_it_start).total_seconds()
                iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                sqp_converged = False
                msg = 'qp_fail'
                if self.verbose: print('QP solution failed')
                break
            dl = l_hat - l

            ls = True
            thresh = 0
            s = np.minimum(thresh, g_i)
            ds = g_i + G_i @ du - s
            if self.merit_function == 'stat_l1':
                constr_vio = g_i - s
                d_stat_norm = float(self.f_dstat_norm(u, l, s, du, dl, x0, up, 0))
                rho = 0.5
                
                if d_stat_norm < 0 and np.sum(constr_vio) > thresh:
                    if self.verbose:
                        print('Case 1: negative directional derivative with constraint violation')
                    mu = -d_stat_norm / ((1-rho)*np.sum(constr_vio))  
                elif d_stat_norm < 0 and np.sum(constr_vio) <= thresh:
                    if self.verbose:
                        print('Case 2: negative directional derivative no constraint violation')
                    mu = 0
                elif d_stat_norm >= 0 and np.sum(constr_vio) > thresh:
                    if self.verbose:
                        print('Case 3: positive directional derivative with constraint violation')
                    mu = d_stat_norm / ((1-rho)*np.sum(constr_vio))  
                elif d_stat_norm >= 0 and np.sum(constr_vio) <= thresh:
                    if self.verbose:
                        print('Case 4: positive directional derivative no constraint violation')
                    mu = 0
                    # u += 0.2*du
                    # l += 0.2*dl
                    # ls = False
            elif self.merit_function == 'stat':
                mu = 0

            # Do line search
            if ls:
                if self.nonmono_ls:
                    u, l, n_qp = self._watchdog_line_search_2(u, du, l, dl, s, ds, x0, up, 
                                    lambda u, l, s: float(self.f_phi(u, l, s, x0, up, mu)),
                                    lambda u, du, l, dl, s, ds: float(self.f_dphi(u, l, s, du, dl, x0, up, mu)),
                                    conv_approx=self.conv_approx)
                    # u, l = self._watchdog_line_search(u, du, l, dl, x0, up, 
                    #                 lambda u, l: float(self.f_phi(u, l, x0, up)),
                    #                 lambda u, du, l, dl: float(self.f_dphi(u, l, du, dl, x0, up)),
                    #                 conv_approx=self.conv_approx)
                    qp_solves += n_qp
                else:
                    u, l, _ = self._line_search_2(u, du, l, dl, s, ds, 
                                    lambda u, l, s: float(self.f_phi(u, l, s, x0, up, mu)),
                                    lambda u, du, l, dl, s, ds: float(self.f_dphi(u, l, s, du, dl, x0, up, mu)))
                    # u, l, _ = self._line_search(u, du, l, dl,
                    #                 lambda u, l: float(self.f_phi(u, l, x0, up)),
                    #                 lambda u, du, l, dl: float(self.f_dphi(u, l, du, dl, x0, up)))     
            
            sqp_it_dur = (datetime.now()-sqp_it_start).total_seconds()
            if self.verbose:
                J = self.f_J(u, x0, up)
                print(f'ego cost: {J[0]}, tar cost: {J[1]}')
                print(f'SQP iteration {sqp_it} time: {sqp_it_dur}')
                print('===================================================')
            
            iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))

            # Convergence via relative tolerance
            if np.linalg.norm(u-u_im1) < xtol/2 and np.linalg.norm(l-l_im1) < ltol/2:
                rel_tol_its += 1
                if rel_tol_its >= self.rel_tol_req and p_feas < xtol:
                    sqp_converged = True
                    msg = 'conv_rel_tol'
                    if self.verbose: print('SQP converged via relative tolerance')
                    break
            else:
                rel_tol_its = 0

            if self.debug_plot:
                self._update_debug_plot(u, x0, up)
                if self.pause_on_plot:
                    pdb.set_trace()

            sqp_it += 1
            if sqp_it >= self.sqp_iters:
                msg = 'max_it'
                sqp_converged = False
                if self.verbose: print('Max SQP iterations reached')
                break
        
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

        solve_dur = (datetime.now()-solve_start).total_seconds()
        print(f'Solve status: {msg}')
        print(f'Solve iters: {sqp_it}')
        print(f'Solve time: {solve_dur}')
        J = self.f_J(u, x0, up)
        print(f'ego cost: {J[0]}, tar cost: {J[1]}')

        solve_info['time'] = solve_dur
        solve_info['num_iters'] = sqp_it
        solve_info['status'] = sqp_converged
        solve_info['cost'] = J
        solve_info['cond'] = cond
        solve_info['iter_data'] = iter_data
        solve_info['msg'] = msg
        solve_info['init'] = init

        if self.debug_plot:
            plt.ioff()

        return solve_info

    def _evaluate(self, u, l, x0, up, hessian=True):
        eval_start = datetime.now()
        x = ca.vertcat(*self.evaluate_dynamics(u, x0))
        A = self.evaluate_jacobian_A(x, u)
        B = self.evaluate_jacobian_B(x, u)
        Du_x = self.f_Du_x(*A, *B)

        g = ca.vertcat(*self.f_Cxu(x, u, up)).full().squeeze()
        H = self.f_Du_C(x, u, up, Du_x)
        q = self.f_q(x, u, up, Du_x).full().squeeze()

        if hessian:
            E = self.evaluate_hessian_E(x, u)
            F = self.evaluate_hessian_F(x, u)
            G = self.evaluate_hessian_G(x, u)
            Q = self.f_Q(x, u, l, up, *A, *B, *E, *F, *G)
            eval_time = (datetime.now()-eval_start).total_seconds()
            if self.verbose:
                print(f'Evaluation time: {eval_time}')
            return Q, q, H, g, x
        else:
            eval_time = (datetime.now()-eval_start).total_seconds()
            if self.verbose:
                print(f'Evaluation time: {eval_time}')
            return q, H, g, x

    def _solve_nonconv_qp(self, Q, q, G, g):
        Q = np.array(Q)
        G = np.array(G)
        Q = (Q + Q.T)/2
        if self.reg > 0:
            Q += self.reg*np.eye(Q.shape[0])
        self.solver_args['p'] = np.concatenate((np.concatenate(np.hsplit(Q, Q.shape[1])).squeeze(),
                                                q,
                                                np.concatenate(np.hsplit(G, G.shape[1])).squeeze(),
                                                g))
        sol = self.solver(**self.solver_args)
        if self.verbose:
            print(self.solver.stats()['return_status'])
        if self.solver.stats()['success']:
            du = sol['x'].toarray().squeeze()
            l_hat = sol['lam_g'].toarray().squeeze()
        else:
            pdb.set_trace()
        return du, l_hat
    
    def _solve_conv_qp(self, Q, q, G, g):
        Q = self._nearestPD(Q)
        if self.reg > 0:
            Q += self.reg*np.eye(Q.shape[0])
        prob = osqp.OSQP()
        prob.setup(P=sp.sparse.csc_matrix(Q),
                q=q,
                A=sp.sparse.csc_matrix(G),
                u=-g,
                polish=True,
                warm_start=True,
                verbose=self.verbose)
        prob.warm_start(x=np.zeros(self.N*self.n_u))
        res = prob.solve()
        du, l_hat = res.x, res.y
        return du, l_hat

    def _build_solver(self):
        # u_0, ..., u_N-1, u_-1
        u_ph = [[ca.MX.sym(f'u_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N+1)] for a in range(self.M)] # Agent inputs
        ua_ph = [ca.vertcat(*u_ph[a][:-1]) for a in range(self.M)] # [u_0^1, ..., u_{N-1}^1, u_0^2, ..., u_{N-1}^2]
        uk_ph = [ca.vertcat(*[u_ph[a][k] for a in range(self.M)]) for k in range(self.N+1)] # [[u_0^1, u_0^2], ..., [u_{N-1}^1, u_{N-1}^2]]

        # Function for evaluating the dynamics function given an input sequence
        xr_ph = [ca.MX.sym('xr_ph_0', self.n_q)] # Initial state
        for k in range(self.N):
            xr_ph.append(self.joint_dynamics.fd(xr_ph[k], uk_ph[k]))
        self.evaluate_dynamics = ca.Function('evaluate_dynamics', [ca.vertcat(*ua_ph), xr_ph[0]], xr_ph, self.options)

        # State sequence placeholders
        x_ph = [ca.MX.sym(f'x_ph_{k}', self.n_q) for k in range(self.N+1)]

        # Function for evaluating the dynamics Jacobians given a state and input sequence
        A, B = [], []
        for k in range(self.N):
            A.append(self.joint_dynamics.fAd(x_ph[k], uk_ph[k]))
            B.append(self.joint_dynamics.fBd(x_ph[k], uk_ph[k]))
        self.evaluate_jacobian_A = ca.Function('evaluate_jacobian_A', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], A, self.options)
        self.evaluate_jacobian_B = ca.Function('evaluate_jacobian_B', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], B, self.options)

        # Placeholders for dynamics Jacobians
        # [Dx0_x1, Dx1_x2, ..., DxN-1_xN]
        A_ph = [ca.MX.sym(f'A_ph_{k}', self.joint_dynamics.sym_Ad.sparsity()) for k in range(self.N)]
        # [Du0_x1, Du1_x2, ..., DuN-1_xN]
        B_ph = [ca.MX.sym(f'B_ph_{k}', self.joint_dynamics.sym_Bd.sparsity()) for k in range(self.N)]

        # Function for evaluating the dynamics Hessians given a state and input sequence
        E, F, G = [], [], []
        for k in range(self.N):
            E += self.joint_dynamics.fEd(x_ph[k], uk_ph[k])
            F += self.joint_dynamics.fFd(x_ph[k], uk_ph[k])
            G += self.joint_dynamics.fGd(x_ph[k], uk_ph[k])
        self.evaluate_hessian_E = ca.Function('evaluate_hessian_E', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], E, self.options)
        self.evaluate_hessian_F = ca.Function('evaluate_hessian_F', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], F, self.options)
        self.evaluate_hessian_G = ca.Function('evaluate_hessian_G', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], G, self.options)

        # Placeholders for dynamics Hessians
        E_ph, F_ph, G_ph = [], [], []
        for k in range(self.N):
            Ek, Fk, Gk = [], [], []
            for i in range(self.n_q):
                Ek.append(ca.MX.sym(f'E{k}_ph_{i}', self.joint_dynamics.sym_Ed[i].sparsity()))
                Fk.append(ca.MX.sym(f'F{k}_ph_{i}', self.joint_dynamics.sym_Fd[i].sparsity()))
                Gk.append(ca.MX.sym(f'G{k}_ph_{i}', self.joint_dynamics.sym_Gd[i].sparsity()))
            E_ph.append(Ek)
            F_ph.append(Fk)
            G_ph.append(Gk)

        Du_x = []
        for k in range(self.N):
            Duk_x = [ca.MX.sym(f'Du{k}_x', ca.Sparsity(self.n_q*(k+1), self.n_u)), B_ph[k]]
            for t in range(k+1, self.N):
                Duk_x.append(A_ph[t] @ Duk_x[-1])
            Du_x.append(ca.vertcat(*Duk_x))
        Du_x = ca.horzcat(*Du_x)
        Du_x = ca.horzcat(*[Du_x[:,self.ua_idxs[a]] for a in range(self.M)])
        self.f_Du_x = ca.Function('f_Du_x', A_ph + B_ph, [Du_x], self.options)

        Du_x_ph = ca.MX.sym('Du_x', Du_x.sparsity())

        # Agent cost functions
        J = [ca.DM.zeros(1) for _ in range(self.M)]
        for a in range(self.M):
            for k in range(self.N):
                J[a] += self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1])
            J[a] += self.costs_sym[a][-1](x_ph[-1])
        self.f_Jxu = ca.Function('f_Jxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]], J)
        # Cost function in batch form
        Ju = self.f_Jxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1])
        self.f_J = ca.Function('f_J', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], Ju, self.options)
        
        # First derivatives of cost function w.r.t. input sequence
        Dx_Jxu = [ca.jacobian(J[a], ca.vertcat(*x_ph)) for a in range(self.M)]
        Du_Jxu = [ca.jacobian(J[a], ca.vertcat(*ua_ph)) for a in range(self.M)]
        Du_J = [(Du_Jxu[a] + Dx_Jxu[a] @ Du_x_ph).T for a in range(self.M)]
        Du_J = [[Du_J[a][int(np.sum(self.num_ua_el[:b])):int(np.sum(self.num_ua_el[:b+1]))] for b in range(self.M)] for a in range(self.M)]
        self.f_Du_J = [ca.Function(f'f_Du_J{a}', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph], Du_J[a], self.options) for a in range(self.M)]
        q = ca.vertcat(*[Du_J[a][a] for a in range(self.M)])
        self.f_q = ca.Function('f_q', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph], [q], self.options)

        # Second derivatves of cost function w.r.t. input sequence using dynamic programming
        Duu_J = []
        for a in range(self.M):
            Duu_Q = []
            Dxu_Q = []
            Dx_Q = [ca.jacobian(self.costs_sym[a][-1](x_ph[-1]), x_ph[-1])]
            Dxx_Q = [ca.jacobian(ca.jacobian(self.costs_sym[a][-1](x_ph[-1]), x_ph[-1]), x_ph[-1])]
            for k in range(self.N-1, -1, -1):
                if k == self.N-1:
                    Jk = self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1])
                else:
                    Jk = self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1]) + self.costs_sym[a][k+1](x_ph[k+1], u_ph[a][k+1], u_ph[a][k])
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
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph))
        self.f_Duu_J = ca.Function('f_Duu_J', in_args, Duu_J, self.options)

        # Duu_J2 = [ca.jacobian(ca.jacobian(Ju[a], ca.vertcat(*ua_ph)), ca.vertcat(*ua_ph)) for a in range(self.M)]
        # self.f_Duu_J2 = ca.Function('f_Duu_J2', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], Duu_J2)

        # Placeholders for gradient of dynamics w.r.t. state and input
        Cs = [[] for _ in range(self.N+1)] # Shared constraints
        Ca = [[[] for _ in range(self.N+1)] for _ in range(self.M)] # Agent specific constraints
        for k in range(self.N):
            # Add shared constraints
            if self.shared_constraints_sym[k] is not None:
                Cs[k].append(self.shared_constraints_sym[k](x_ph[k], uk_ph[k], uk_ph[k-1]))
            if len(Cs[k]) > 0:
                Cs[k] = ca.vertcat(*Cs[k])
                self.n_cs[k] = Cs[k].shape[0]
            else:
                Cs[k] = ca.DM()
            # Add agent constraints
            for a in range(self.M):
                if self.constraints_sym[a][k] is not None:
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
            Cs[-1].append(self.shared_constraints_sym[-1](x_ph[-1]))
        if len(Cs[-1]) > 0:
            Cs[-1] = ca.vertcat(*Cs[-1])
            self.n_cs[-1] = Cs[-1].shape[0]
        else:
            Cs[-1] = ca.DM()
        # Add agent constraints
        for a in range(self.M):
            if self.constraints_sym[a][-1] is not None:
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
        self.f_Cxu = ca.Function('f_Cxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]], C, self.options)
        
        # Constraint function in batch form
        Cu = self.f_Cxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1])
        self.f_C = ca.Function('f_C', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], Cu, self.options)
        
        # First derivatives of constraints w.r.t. input sequence
        Dx_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*x_ph))
        Du_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*ua_ph))
        Du_C = Du_Cxu + Dx_Cxu @ Du_x_ph
        self.f_Du_C = ca.Function('f_Du_C', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph], [Du_C], self.options)
        
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
                Duu = ca.MX.sym(f'Duu_C{k}_{j}', ca.Sparsity(self.n_u*self.N, self.n_u*self.N))
                Duu[:Duu_Cj[-1].size1(),:Duu_Cj[-1].size2()] = Duu_Cj[-1]
                Duu = ca.horzcat(*[Duu[:,self.ua_idxs[a]] for a in range(self.M)])
                Duu = ca.vertcat(*[Duu[self.ua_idxs[a],:] for a in range(self.M)])
                Duu_C.append(Duu)

        l_ph = ca.MX.sym(f'l_ph', np.sum(self.n_c))
        lDuu_C = 0
        for j in range(np.sum(self.n_c)):
            lDuu_C += l_ph[j] * Duu_C[j]
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), l_ph, uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph))
        self.f_lDuu_C = ca.Function('f_lDuu_C', in_args, [lDuu_C], self.options)

        # Hessian of the Lagrangian
        Q = ca.vertcat(*[Duu_J[a][int(np.sum(self.num_ua_el[:a])):int(np.sum(self.num_ua_el[:a+1])),:] for a in range(self.M)]) + lDuu_C
        self.f_Q = ca.Function('f_Q', in_args, [Q], self.options)

        # Symbolic Hessian of Lagrangian
        L = [Ju[a] + ca.dot(l_ph, ca.vertcat(*Cu)) for a in range(self.M)]
        Du_L = [[ca.jacobian(L[a], ua_ph[b]).T for b in range(self.M)] for a in range(self.M)]
        Duu_L = [[ca.jacobian(Du_L[a][b], ca.vertcat(*ua_ph)) for b in range(self.M)] for a in range(self.M)]

        Q2 = ca.vertcat(*[Duu_L[a][a] for a in range(self.M)]) 
        self.f_Q2 = ca.Function('f_Q', [ca.vertcat(*ua_ph), l_ph, xr_ph[0], uk_ph[-1]], [Q2])

        # Merit function
        du_ph = [[ca.MX.sym(f'du_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N)] for a in range(self.M)] # Agent inputs
        dua_ph = [ca.vertcat(*du_ph[a]) for a in range(self.M)] # Stack input sequences by agent
        dl_ph = ca.MX.sym(f'dl_ph', np.sum(self.n_c))
        s_ph = ca.MX.sym(f's_ph', np.sum(self.n_c))
        ds_ph = ca.MX.sym(f'ds_ph', np.sum(self.n_c))
        mu_ph = ca.MX.sym('mu_ph', 1)

        # d_ph = ca.MX.sym('d_ph', self.n_u*self.N)
        # q_ph = ca.MX.sym('q_ph', self.n_u*self.N)
        # Q_ph = ca.MX.sym('Q_ph', Q.sparsity())
        # g_ph = ca.MX.sym('g_ph', np.sum(self.n_c))
        # H_ph = ca.MX.sym('H_ph', Du_C.sparsity())

        # stat = ca.vertcat(q + Du_C.T @ l_ph, ca.dot(l_ph, ca.vertcat(*C)))
        # stat_norm = (1/2)*ca.bilin(ca.DM.eye(stat.size1()), stat, stat)
        # dstat_norm = (q + Du_C.T @ l_ph).T @ ca.horzcat(Q, Du_C.T) @ ca.vertcat(*dua_ph, dl_ph) \
        #                 + ca.dot(l_ph, ca.vertcat(*C))*(l_ph.T @ Du_C @ ca.vertcat(*dua_ph) + ca.dot(dl_ph, ca.vertcat(*C)))

        # vio = mu_ph*ca.sum1(ca.vertcat(*C)-s_ph)
        # dvio = -mu_ph*ca.sum1(ca.vertcat(*C)-s_ph)

        # phi_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), l_ph, s_ph, mu_ph]
        # self.f_phi = ca.Function('f_phi', phi_args, [stat_norm + vio])
        # dphi_args = phi_args + [ca.vertcat(*dua_ph), dl_ph]
        # self.f_dphi = ca.Function('f_dphi', dphi_args, [dstat_norm + dvio])
        # self.f_dstat_norm = ca.Function('f_dstat_norm', dphi_args, [dstat_norm])

        # Stationarity plus constraint violation
        stat2 = ca.vertcat(*[Du_L[a][a] for a in range(self.M)], ca.dot(l_ph, ca.vertcat(*Cu)))
        stat_norm2 = (1/2)*ca.bilin(ca.DM.eye(stat2.size1()), stat2, stat2)
        dstat_norm2 = ca.jtimes(stat_norm2, ca.vertcat(*ua_ph, l_ph), ca.vertcat(*dua_ph, dl_ph), False)
        vio2 = mu_ph*ca.sum1(ca.vertcat(*Cu)-s_ph)
        dvio2 = -mu_ph*ca.sum1(ca.vertcat(*Cu)-s_ph)

        phi2_args = [ca.vertcat(*ua_ph), l_ph, s_ph, xr_ph[0], uk_ph[-1], mu_ph]
        dphi2_args = [ca.vertcat(*ua_ph), l_ph, s_ph, ca.vertcat(*dua_ph), dl_ph, xr_ph[0], uk_ph[-1], mu_ph]
        if self.merit_function == 'stat_l1':
            self.f_phi = ca.Function('f_phi2', phi2_args, [stat_norm2 + vio2])
            self.f_dphi = ca.Function('f_dphi2', dphi2_args, [dstat_norm2 + dvio2])
        elif self.merit_function == 'stat':
            self.f_phi = ca.Function('f_phi2', phi2_args, [stat_norm2])
            self.f_dphi = ca.Function('f_dphi2', dphi2_args, [dstat_norm2])
        else:
            raise(ValueError(f'Merit function option {self.merit_function} not recognized'))

        self.f_dstat_norm = ca.Function('f_dstat_norm2', dphi2_args, [dstat_norm2])

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
                print(f'- Generating C code for solver {self.solver_name} at {str(gen_path)}')
            generator.generate()
            # Compile into shared object
            so_path = gen_path.joinpath(self.so_file_name)
            command = f'gcc -fPIC -shared -{self.opt_flag} {c_path} -o {so_path}'
            if self.verbose:
                print(f'- Compiling shared object {so_path} from {c_path}')
                print(f'- Executing "{command}"')
            pdb.set_trace()
            os.system(command)
            # Swtich back to working directory
            os.chdir(cur_dir)
            install_dir = self.install()

            # Load solver
            self._load_solver(str(install_dir.joinpath(self.so_file_name)))

    def _load_solver(self, solver_path=None):
        if solver_path is None:
            solver_path = str(pathlib.Path(self.solver_dir, self.so_file_name).expanduser())
        if self.verbose:
            print(f'- Loading solver from {solver_path}')
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

    def _line_search(self, u, du, l, dl, merit, d_merit):
        phi = merit(u, l)
        dphi = d_merit(u, du, l, dl)
        if dphi > 0:
            if self.verbose:
                print(f'- Line search directional derivative is positive: {dphi}')
        alpha, conv = 1.0, False
        for i in range(self.line_search_iters):
            u_trial = u + alpha*du
            l_trial = l + alpha*dl
            phi_trial = merit(u_trial, l_trial)
            if self.verbose:
                print(f'- Line search iteration: {i} | merit gap: {phi_trial-(phi + self.beta*alpha*dphi):.4e} | a: {alpha:.4e}')
            if phi_trial <= phi + self.beta*alpha*dphi:
                conv = True
                break
            else:
                alpha *= self.tau
        if not conv:
            if self.verbose:
                print('- Max iterations reached, line search did not succeed')
            # pdb.set_trace()
        return u_trial, l_trial, phi_trial

    def _line_search_2(self, u, du, l, dl, s, ds, merit, d_merit):
        phi = merit(u, l, s)
        dphi = d_merit(u, du, l, dl, s, ds)
        if dphi > 0:
            if self.verbose:
                print(f'- Line search directional derivative is positive: {dphi}')
        alpha, conv = 1.0, False
        for i in range(self.line_search_iters):
            u_trial = u + alpha*du
            l_trial = l + alpha*dl
            s_trial = s + alpha*ds
            phi_trial = merit(u_trial, l_trial, s_trial)
            # dphi_trial = d_merit(u_trial, du, l_trial, dl, s_trial, ds)
            if self.verbose:
                print(f'- Line search iteration: {i} | merit gap: {phi_trial-(phi + self.beta*alpha*dphi):.4e} | a: {alpha:.4e}')
            # if phi_trial <= phi + self.beta*alpha*dphi and dphi_trial >= 2*self.beta*dphi:
            if phi_trial <= phi + self.beta*alpha*dphi:
                conv = True
                break
            else:
                alpha *= self.tau
        if not conv:
            if self.verbose:
                print('- Max iterations reached, line search did not succeed')
            # pdb.set_trace()
        return u_trial, l_trial, phi_trial

    def _watchdog_line_search(self, u_k, du_k, l_k, dl_k, x0, up, merit, d_merit, conv_approx=False):
        if self.verbose:
            print('===================================================')
            print('Watchdog step acceptance routine')
        t_hat = 7 # Number of steps where we search for sufficient merit decrease
        phi_log = []

        phi_k = merit(u_k, l_k)
        phi_log.append(phi_k)
        dphi_k = d_merit(u_k, du_k, l_k, dl_k)
        if dphi_k > 0:
            if self.verbose:
                print(f'Time k: Directional derivative is positive: {dphi_k}')

        # Take relaxed (full) step
        u_kp1 = u_k + du_k
        l_kp1 = l_k + dl_k
        phi_kp1 = merit(u_kp1, l_kp1)
        phi_log.append(phi_kp1)

        # Check for sufficient decrease w.r.t. time k
        if self.verbose:
            print(f'Time k+1:')
        if phi_kp1 <= phi_k + self.beta*dphi_k:
            if self.verbose:
                print(f'Sufficient decrease achieved')
            return u_kp1, l_kp1
        if self.verbose:
            print(f'Insufficient decrease in merit')

        # Check for sufficient decrease in the next t_hat steps
        u_t, l_t, phi_t = u_kp1, l_kp1, phi_kp1
        for t in range(t_hat):
            if self.verbose:
                print(f'Time k+{t+2}:')
            # Compute step at time t
            Q_t, q_t, G_t, g_t, x_t = self._evaluate(u_t, l_t, x0, up)
            if conv_approx:
                du_t, l_hat = self._solve_conv_qp(Q_t, q_t, G_t, g_t)
            else:
                du_t, l_hat = self._solve_nonconv_qp(Q_t, q_t, G_t, g_t)
            dl_t = l_hat - l_t

            # Do line search
            u_tp1, l_tp1, phi_tp1 = self._line_search(u_t, du_t, l_t, dl_t, merit, d_merit)
            phi_log.append(phi_tp1)
            if self.verbose:
                print(phi_log)
            
            # Check for sufficient decrease w.r.t. time 0
            if phi_t <= phi_k or phi_tp1 <= phi_k + self.beta*dphi_k:
                if self.verbose:
                    print(f'Sufficient decrease achieved')
                return u_tp1, l_tp1
            
            # Update for next time step
            u_t, l_t, phi_t = u_tp1, l_tp1, phi_tp1
        
        if phi_tp1 > phi_k:
            if self.verbose:
                print(f'No decrease in merit, returning to search along step at time k')
            u_kp1, l_kp1, phi_kp1 = self._line_search(u_k, du_k, l_k, dl_k, merit, d_merit)
            return u_kp1, l_kp1
        else:
            if self.verbose:
                print(f'Insufficient decrease in merit')
            Q_tp1, q_tp1, G_tp1, g_tp1, x_tp1 = self._evaluate(u_tp1, l_tp1, x0, up)
            if conv_approx:
                du_tp1, l_hat = self._solve_conv_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
            else:
                du_tp1, l_hat = self._solve_nonconv_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
            dl_tp1 = l_hat - l_tp1

            u_tp2, l_tp2, phi_tp2 = self._line_search(u_tp1, du_tp1, l_tp1, dl_tp1, merit, d_merit)
            return u_tp2, l_tp2

    def _watchdog_line_search_2(self, u_k, du_k, l_k, dl_k, s_k, ds_k, x0, up, merit, d_merit, conv_approx=False):
        if self.verbose:
            print('===================================================')
            print('Watchdog step acceptance routine')
        qp_solves = 0
        t_hat = 7 # Number of steps where we search for sufficient merit decrease
        phi_log = []

        phi_k = merit(u_k, l_k, s_k)
        phi_log.append(phi_k)
        dphi_k = d_merit(u_k, du_k, l_k, dl_k, s_k, ds_k)
        if dphi_k > 0:
            if self.verbose:
                print(f'Time k: Directional derivative is positive: {dphi_k}')

        # Take relaxed (full) step
        u_kp1 = u_k + du_k
        l_kp1 = l_k + dl_k
        s_kp1 = s_k + ds_k
        phi_kp1 = merit(u_kp1, l_kp1, s_kp1)
        phi_log.append(phi_kp1)

        # Check for sufficient decrease w.r.t. time k
        if self.verbose:
            print(f'Time k+1:')
        if phi_kp1 <= phi_k + self.beta*dphi_k:
            if self.verbose:
                print(f'Sufficient decrease achieved')
            return u_kp1, l_kp1, qp_solves
        if self.verbose:
            print(f'Insufficient decrease in merit')

        # Check for sufficient decrease in the next t_hat steps
        u_t, l_t, phi_t = u_kp1, l_kp1, phi_kp1
        for t in range(t_hat):
            if self.verbose:
                print(f'Time k+{t+2}:')
            # Compute step at time t
            Q_t, q_t, G_t, g_t, x_t = self._evaluate(u_t, l_t, x0, up)
            if conv_approx:
                du_t, l_hat = self._solve_conv_qp(Q_t, q_t, G_t, g_t)
            else:
                du_t, l_hat = self._solve_nonconv_qp(Q_t, q_t, G_t, g_t)
            qp_solves += 1
            if None in l_hat:
                if self.verbose:
                    print(f'QP failed, returning to search along step at time k')
                u_kp1, l_kp1, phi_kp1 = self._line_search_2(u_k, du_k, l_k, dl_k, s_k, ds_k, merit, d_merit)
                return u_kp1, l_kp1, qp_solves
            dl_t = l_hat - l_t

            s_t = np.minimum(1e-7, g_t)
            ds_t = g_t + G_t @ du_t - s_t

            # Do line search
            u_tp1, l_tp1, phi_tp1 = self._line_search_2(u_t, du_t, l_t, dl_t, s_t, ds_t, merit, d_merit)
            phi_log.append(phi_tp1)
            if self.verbose:
                print(phi_log)
            
            # Check for sufficient decrease w.r.t. time 0
            if phi_t <= phi_k or phi_tp1 <= phi_k + self.beta*dphi_k:
                if self.verbose:
                    print(f'Sufficient decrease achieved')
                return u_tp1, l_tp1, qp_solves
            
            # Update for next time step
            u_t, l_t, phi_t = u_tp1, l_tp1, phi_tp1
        
        if phi_tp1 > phi_k:
            if self.verbose:
                print(f'No decrease in merit, returning to search along step at time k')
            u_kp1, l_kp1, phi_kp1 = self._line_search_2(u_k, du_k, l_k, dl_k, s_k, ds_k, merit, d_merit)
            return u_kp1, l_kp1, qp_solves
        else:
            if self.verbose:
                print(f'Insufficient decrease in merit')
            Q_tp1, q_tp1, G_tp1, g_tp1, x_tp1 = self._evaluate(u_tp1, l_tp1, x0, up)
            if conv_approx:
                du_tp1, l_hat = self._solve_conv_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
            else:
                du_tp1, l_hat = self._solve_nonconv_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
            if None in l_hat:
                if self.verbose:
                    print(f'QP failed, returning to search along step at time k')
                u_kp1, l_kp1, phi_kp1 = self._line_search_2(u_k, du_k, l_k, dl_k, s_k, ds_k, merit, d_merit)
                return u_kp1, l_kp1, qp_solves
            qp_solves += 1
            dl_tp1 = l_hat - l_tp1

            s_tp1 = np.minimum(1e-7, g_tp1)
            ds_tp1 = g_tp1 + G_tp1 @ du_tp1 - s_tp1

            u_tp2, l_tp2, phi_tp2 = self._line_search_2(u_tp1, du_tp1, l_tp1, dl_tp1, s_tp1, ds_tp1, merit, d_merit)
            return u_tp2, l_tp2, qp_solves

    def _nearestPD(self, A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        if not np.allclose(A, A.T):
            B = (A + A.T) / 2
            _, s, V = np.linalg.svd(B)

            H = V.T @ np.diag(s) @ V

            A2 = (B + H) / 2

            A3 = (A2 + A2.T) / 2
        else:
            A3 = A

        if self._isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not self._isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    def _isPD(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def _update_debug_plot(self, u, x0, up):
        q_bar = np.array(self.evaluate_dynamics(u, x0)).squeeze()
        ua_bar = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_el[:a]))
            ei = int(np.sum(self.num_ua_el[:a])+self.num_ua_el[a])
            ua_bar.append(u[si:ei].reshape((self.N, self.num_ua_d[a])))
        u_bar = np.hstack(ua_bar)
        if not self.local_pos:
            self.l1_xy.set_data(q_bar[:,0], q_bar[:,1])
            self.l2_xy.set_data(q_bar[:,0+self.joint_dynamics.dynamics_models[0].n_q], q_bar[:,1+self.joint_dynamics.dynamics_models[0].n_q])
        else:
            raise NotImplementedError('Conversion from local to global pos has not been implemented for debug plot')
        self.ax_xy.set_aspect('equal')
        J = self.f_J(u, x0, up)
        self.ax_xy.set_title(f'ego cost: {J[0]}, tar cost: {J[1]}')
        self.l1_a.set_data(np.arange(self.N), u_bar[:,0])
        self.l1_s.set_data(np.arange(self.N), u_bar[:,1])
        self.l2_a.set_data(np.arange(self.N), u_bar[:,2])
        self.l2_s.set_data(np.arange(self.N), u_bar[:,3])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == '__main__':
    pass