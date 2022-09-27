#!/usr/bin python3

import numpy as np
import scipy as sp
import casadi as ca

import copy
import pdb
import time
import itertools

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
                       params=DGSQPParams(),
                       print_method=print,
                       xy_plot=None):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a
        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method

        self.N                  = params.N

        self.reg                = params.reg
        self.line_search_iters  = params.line_search_iters
        self.nonmono_ls         = params.nonmono_ls
        self.sqp_iters          = params.sqp_iters
        self.merit_function     = params.merit_function

        # Convergence tolerance for SQP
        self.p_tol              = params.p_tol
        self.d_tol              = params.d_tol
        self.rel_tol_req        = 5

        # Line search parameters
        self.beta               = params.beta
        self.tau                = params.tau

        self.verbose            = params.verbose
        self.save_iter_data     = params.save_iter_data
        if params.time_limit is None:
            self.time_limit = np.inf
        else:
            self.time_limit = params.time_limit

        self.solver_name        = params.solver_name

        self.debug_plot         = params.debug_plot
        self.pause_on_plot      = params.pause_on_plot
        self.local_pos          = params.local_pos

        if self.debug_plot:
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            self.ax_a = self.fig.add_subplot(2,2,2)
            self.ax_s = self.fig.add_subplot(2,2,4)
            if xy_plot is not None:
                xy_plot(self.ax_xy)
            self.colors = ['b', 'g', 'r', 'm', 'c']
            self.l_xy, self.l_a, self.l_s = [], [], []
            for i in range(self.M):
                self.l_xy.append(self.ax_xy.plot([], [], f'{self.colors[i]}o')[0])
                self.l_a.append(self.ax_a.plot([], [], f'-{self.colors[i]}o')[0])
                self.l_s.append(self.ax_s.plot([], [], f'-{self.colors[i]}o')[0])
            self.ax_a.set_ylabel('accel')
            self.ax_s.set_ylabel('steering')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

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

        # Construct QP solver
        solver = 'osqp'
        solver_opts = dict(error_on_fail=False, osqp=dict(polish=True, verbose=self.verbose))
        prob = {'h': self.f_Q.sparsity_out(0), 'a': self.f_Du_C.sparsity_out(0)}
        self.solver = ca.conic('qp', solver, prob, solver_opts)
        self.dual_name = 'lam_a'
        self.initialized = True

    def _solve_qp(self, Q, q, G, g, x0=None):
        t = time.time()
        Q = self._nearestPD(Q)
        if self.reg > 0:
            Q += self.reg*np.eye(Q.shape[0])
        if x0 is None:
            x0 = np.zeros(self.N*self.n_u)

        try:
            sol = self.solver(h=Q, g=q, a=G, uba=-g, x0=x0)
            if self.verbose:
                self.print_method(self.solver.stats()['return_status'])
                self.print_method(f'Total QP solve time: {time.time()-t}')
            du = sol['x'].toarray().squeeze()
            l_hat = sol[self.dual_name].toarray().squeeze()
        except:
            du = [None]
            l_hat = [None]
        
        return du, l_hat

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

    def step(self, states: List[VehicleState]):
        info = self.solve(states)

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

    def solve(self, states: List[VehicleState]):
        solve_info = {}
        solve_start = time.time()
        self.u_prev = np.zeros(self.n_u)

        u = copy.copy(self.u_ws)
        up = copy.copy(self.u_prev)

        x0 = self.joint_dynamics.state2q(states)

        # Warm start dual variables
        q, G, _, _ = self._evaluate(u, None, x0, up, hessian=False)
        G = G.sparse()
        l = np.maximum(0, -sp.sparse.linalg.lsqr(G @ G.T, G @ q)[0])
        if l is None:
            l = np.zeros(np.sum(self.n_c))
        init = dict(u=u, l=l)

        if self.debug_plot:
            self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up))
            if self.pause_on_plot:
                pdb.set_trace()

        sqp_converged = False
        rel_tol_its = 0
        sqp_it = 0
        iter_data = []
        self.print_method(self.solver_name)
        while True:
            qp_solves = 0
            sqp_it_start = time.time()
            if self.verbose:
                self.print_method('===================================================')
                self.print_method(f'DGSQP iteration: {sqp_it}')

            if self.debug_plot:
                self._update_debug_plot(copy.copy(u), copy.copy(x0), copy.copy(up))
                if self.pause_on_plot:
                    pdb.set_trace()
                    
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
                self.print_method(f'SQP iteration {sqp_it}')
                self.print_method(f'p feas: {p_feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e}')
            if stat > 1e5:
                sqp_it_dur = time.time() - sqp_it_start
                if self.save_iter_data:
                    iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                if self.verbose: self.print_method('SQP diverged')
                msg = 'diverged'
                sqp_converged = False
                break
            if p_feas < xtol and comp < ltol and stat < ltol:
                sqp_it_dur = time.time() - sqp_it_start
                if self.save_iter_data:
                    iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                sqp_converged = True
                msg = 'conv_abs_tol'
                if self.verbose: self.print_method('SQP converged via optimality conditions')
                break
            
            # Compute SQP primal dual step
            du, l_hat = self._solve_qp(Q_i, q_i, G_i, g_i)
            qp_solves = 1
            if None in du:
                sqp_it_dur = time.time() - sqp_it_start
                if self.save_iter_data:
                    iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))
                sqp_converged = False
                msg = 'qp_fail'
                if self.verbose: self.print_method('QP solution failed')
                break
            dl = l_hat - l

            ls = True
            s = np.minimum(0, g_i)
            ds = g_i + G_i @ du - s
            mu = self._get_mu(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i)

            # Do line search
            if ls:
                if self.nonmono_ls:
                    u, l, n_qp = self._watchdog_line_search(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i, 
                                    lambda u, l, hessian: self._evaluate(u, l, x0, up, hessian),
                                    lambda u, l, s, q, G, g: float(self.f_phi(l, s, q, G, g, mu)),
                                    lambda u, du, l, dl, s, ds, Q, q, G, g: float(self.f_dphi(du, l, dl, s, Q, q, G, g, mu)))
                    qp_solves += n_qp
                else:
                    u, l, _ = self._line_search(u, du, l, dl, s, ds, Q_i, q_i, G_i, g_i, 
                                    lambda u, l, hessian: self._evaluate(u, l, x0, up, hessian),
                                    lambda u, l, s, q, G, g: float(self.f_phi(l, s, q, G, g, mu)),
                                    lambda u, du, l, dl, s, ds, Q, q, G, g: float(self.f_dphi(du, l, dl, s, Q, q, G, g, mu)))
            else:
                u += 0.01*du
                l += 0.01*dl

            sqp_it_dur = time.time() - sqp_it_start
            if self.verbose:
                J = self.f_J(u, x0, up)
                self.print_method(str(J))
                self.print_method(f'SQP iteration {sqp_it} time: {sqp_it_dur}')
                self.print_method('===================================================')
            
            iter_data.append(dict(cond=cond, u_sol=u, l_sol=l, qp_solves=qp_solves, it_time=sqp_it_dur))

            # Convergence via relative tolerance
            if np.linalg.norm(u-u_im1) < xtol/2 and np.linalg.norm(l-l_im1) < ltol/2:
                rel_tol_its += 1
                if rel_tol_its >= self.rel_tol_req and p_feas < xtol:
                    sqp_converged = True
                    msg = 'conv_rel_tol'
                    if self.verbose: self.print_method('SQP converged via relative tolerance')
                    break
            else:
                rel_tol_its = 0

            sqp_it += 1
            if sqp_it >= self.sqp_iters:
                msg = 'max_it'
                sqp_converged = False
                if self.verbose: self.print_method('Max SQP iterations reached')
                break
            if time.time() - solve_start > self.time_limit:
                msg = 'time_limit'
                sqp_converged = False
                if self.verbose: self.print_method('Time limit reached')
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

        solve_dur = time.time() - solve_start
        self.print_method(f'Solve status: {msg}')
        self.print_method(f'Solve iters: {sqp_it}')
        self.print_method(f'Solve time: {solve_dur}')
        J = self.f_J(u, x0, up)
        self.print_method(str(J))

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
        eval_start = time.time()
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
            eval_time = time.time() - eval_start
            if self.verbose:
                self.print_method(f'Evaluation time: {eval_time}')
            return Q, q, H, g, x
        else:
            eval_time = time.time() - eval_start
            if self.verbose:
                self.print_method(f'Evaluation time: {eval_time}')
            return q, H, g, x

    def _get_mu(self, u, du, l, dl, s, ds, Q, q, G, g):
        thresh = 0
        if self.merit_function == 'stat_l1':
            constr_vio = g - s
            d_stat_norm = float(self.f_dstat_norm(du, l, dl, s, Q, q, G, g, 0))
            rho = 0.5
            
            if d_stat_norm < 0 and np.sum(constr_vio) > thresh:
                if self.verbose:
                    self.print_method('Case 1: negative directional derivative with constraint violation')
                mu = -d_stat_norm / ((1-rho)*np.sum(constr_vio))  
            elif d_stat_norm < 0 and np.sum(constr_vio) <= thresh:
                if self.verbose:
                    self.print_method('Case 2: negative directional derivative no constraint violation')
                mu = 0
            elif d_stat_norm >= 0 and np.sum(constr_vio) > thresh:
                if self.verbose:
                    self.print_method('Case 3: positive directional derivative with constraint violation')
                mu = d_stat_norm / ((1-rho)*np.sum(constr_vio))  
            elif d_stat_norm >= 0 and np.sum(constr_vio) <= thresh:
                if self.verbose:
                    self.print_method('Case 4: positive directional derivative no constraint violation')
                mu = 0
        elif self.merit_function == 'stat':
            mu = 0
        
        return mu

    def _build_solver(self):
        # u_0, ..., u_N-1, u_-1
        u_ph = [[ca.SX.sym(f'u_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N+1)] for a in range(self.M)] # Agent inputs
        ua_ph = [ca.vertcat(*u_ph[a][:-1]) for a in range(self.M)] # [u_0^1, ..., u_{N-1}^1, u_0^2, ..., u_{N-1}^2]
        uk_ph = [ca.vertcat(*[u_ph[a][k] for a in range(self.M)]) for k in range(self.N+1)] # [[u_0^1, u_0^2], ..., [u_{N-1}^1, u_{N-1}^2]]

        # Function for evaluating the dynamics function given an input sequence
        xr_ph = [ca.SX.sym('xr_ph_0', self.n_q)] # Initial state
        for k in range(self.N):
            xr_ph.append(self.joint_dynamics.fd(xr_ph[k], uk_ph[k]))
        self.evaluate_dynamics = ca.Function('evaluate_dynamics', [ca.vertcat(*ua_ph), xr_ph[0]], xr_ph)

        # State sequence placeholders
        x_ph = [ca.SX.sym(f'x_ph_{k}', self.n_q) for k in range(self.N+1)]

        # Function for evaluating the dynamics Jacobians given a state and input sequence
        A, B = [], []
        for k in range(self.N):
            A.append(self.joint_dynamics.fAd(x_ph[k], uk_ph[k]))
            B.append(self.joint_dynamics.fBd(x_ph[k], uk_ph[k]))
        self.evaluate_jacobian_A = ca.Function('evaluate_jacobian_A', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], A)
        self.evaluate_jacobian_B = ca.Function('evaluate_jacobian_B', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph)], B)

        # Placeholders for dynamics Jacobians
        # [Dx0_x1, Dx1_x2, ..., DxN-1_xN]
        A_ph = [ca.SX.sym(f'A_ph_{k}', self.joint_dynamics.sym_Ad.sparsity()) for k in range(self.N)]
        # [Du0_x1, Du1_x2, ..., DuN-1_xN]
        B_ph = [ca.SX.sym(f'B_ph_{k}', self.joint_dynamics.sym_Bd.sparsity()) for k in range(self.N)]

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
                Ek.append(ca.SX.sym(f'E{k}_ph_{i}', self.joint_dynamics.sym_Ed[i].sparsity()))
                Fk.append(ca.SX.sym(f'F{k}_ph_{i}', self.joint_dynamics.sym_Fd[i].sparsity()))
                Gk.append(ca.SX.sym(f'G{k}_ph_{i}', self.joint_dynamics.sym_Gd[i].sparsity()))
            E_ph.append(Ek)
            F_ph.append(Fk)
            G_ph.append(Gk)

        Du_x = []
        for k in range(self.N):
            Duk_x = [ca.SX.sym(f'Du{k}_x', ca.Sparsity(self.n_q*(k+1), self.n_u)), B_ph[k]]
            for t in range(k+1, self.N):
                Duk_x.append(A_ph[t] @ Duk_x[-1])
            Du_x.append(ca.vertcat(*Duk_x))
        Du_x = ca.horzcat(*Du_x)
        Du_x = ca.horzcat(*[Du_x[:,self.ua_idxs[a]] for a in range(self.M)])
        self.f_Du_x = ca.Function('f_Du_x', A_ph + B_ph, [Du_x])

        Du_x_ph = ca.SX.sym('Du_x', Du_x.sparsity())

        # Agent cost functions
        J = [ca.DM.zeros(1) for _ in range(self.M)]
        for a in range(self.M):
            for k in range(self.N):
                J[a] += self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1])
            J[a] += self.costs_sym[a][-1](x_ph[-1])
        self.f_Jxu = ca.Function('f_Jxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]], J)
        # Cost function in batch form
        Ju = self.f_Jxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1])
        self.f_J = ca.Function('f_J', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], Ju)
        
        # First derivatives of cost function w.r.t. input sequence
        Dx_Jxu = [ca.jacobian(J[a], ca.vertcat(*x_ph)) for a in range(self.M)]
        Du_Jxu = [ca.jacobian(J[a], ca.vertcat(*ua_ph)) for a in range(self.M)]
        Du_J = [(Du_Jxu[a] + Dx_Jxu[a] @ Du_x_ph).T for a in range(self.M)]
        Du_J = [[Du_J[a][int(np.sum(self.num_ua_el[:b])):int(np.sum(self.num_ua_el[:b+1]))] for b in range(self.M)] for a in range(self.M)]
        self.f_Du_J = [ca.Function(f'f_Du_J{a}', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph], Du_J[a]) for a in range(self.M)]
        q = ca.vertcat(*[Du_J[a][a] for a in range(self.M)])
        self.f_q = ca.Function('f_q', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph], [q])

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
        self.f_Duu_J = ca.Function('f_Duu_J', in_args, Duu_J)

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
        self.f_Cxu = ca.Function('f_Cxu', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1]], C)
        
        # Constraint function in batch form
        Cu = self.f_Cxu(ca.vertcat(*xr_ph), ca.vertcat(*ua_ph), uk_ph[-1])
        self.f_C = ca.Function('f_C', [ca.vertcat(*ua_ph), xr_ph[0], uk_ph[-1]], Cu)
        
        # First derivatives of constraints w.r.t. input sequence
        Dx_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*x_ph))
        Du_Cxu = ca.jacobian(ca.vertcat(*C), ca.vertcat(*ua_ph))
        Du_C = Du_Cxu + Dx_Cxu @ Du_x_ph
        self.f_Du_C = ca.Function('f_Du_C', [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), uk_ph[-1], Du_x_ph], [Du_C])
        
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
                Duu = ca.SX.sym(f'Duu_C{k}_{j}', ca.Sparsity(self.n_u*self.N, self.n_u*self.N))
                Duu[:Duu_Cj[-1].size1(),:Duu_Cj[-1].size2()] = Duu_Cj[-1]
                Duu = ca.horzcat(*[Duu[:,self.ua_idxs[a]] for a in range(self.M)])
                Duu = ca.vertcat(*[Duu[self.ua_idxs[a],:] for a in range(self.M)])
                Duu_C.append(Duu)

        l_ph = ca.SX.sym(f'l_ph', np.sum(self.n_c))
        lDuu_C = 0
        for j in range(np.sum(self.n_c)):
            lDuu_C += l_ph[j] * Duu_C[j]
        in_args = [ca.vertcat(*x_ph), ca.vertcat(*ua_ph), l_ph, uk_ph[-1]] \
                    + A_ph + B_ph \
                    + list(itertools.chain(*E_ph)) \
                    + list(itertools.chain(*F_ph)) \
                    + list(itertools.chain(*G_ph))
        self.f_lDuu_C = ca.Function('f_lDuu_C', in_args, [lDuu_C])

        # Hessian of the Lagrangian
        Q = ca.vertcat(*[Duu_J[a][int(np.sum(self.num_ua_el[:a])):int(np.sum(self.num_ua_el[:a+1])),:] for a in range(self.M)]) + lDuu_C
        self.f_Q = ca.Function('f_Q', in_args, [Q])

        # Merit function
        du_ph = [[ca.SX.sym(f'du_{a}_ph_{k}', self.joint_dynamics.dynamics_models[a].n_u) for k in range(self.N)] for a in range(self.M)] # Agent inputs
        dua_ph = [ca.vertcat(*du_ph[a]) for a in range(self.M)] # Stack input sequences by agent
        dl_ph = ca.SX.sym(f'dl_ph', np.sum(self.n_c))
        s_ph = ca.SX.sym(f's_ph', np.sum(self.n_c))
        ds_ph = ca.SX.sym(f'ds_ph', np.sum(self.n_c))
        mu_ph = ca.SX.sym('mu_ph', 1)

        q_ph = ca.SX.sym('q_ph', self.n_u*self.N)
        Q_ph = ca.SX.sym('Q_ph', Q.sparsity())
        g_ph = ca.SX.sym('g_ph', np.sum(self.n_c))
        H_ph = ca.SX.sym('H_ph', Du_C.sparsity())

        stat = ca.vertcat(q_ph + H_ph.T @ l_ph, ca.dot(l_ph, g_ph))
        stat_norm = (1/2)*ca.sumsqr(stat)
        dstat_norm = (q_ph + H_ph.T @ l_ph).T @ ca.horzcat(Q_ph, H_ph.T) @ ca.vertcat(*dua_ph, dl_ph) \
                        + ca.dot(l_ph, g_ph)*(l_ph.T @ H_ph @ ca.vertcat(*dua_ph) + ca.dot(dl_ph, g_ph))
        vio = mu_ph*ca.sum1(g_ph-s_ph)
        dvio = -mu_ph*ca.sum1(g_ph-s_ph)

        phi_args = [l_ph, s_ph, q_ph, H_ph, g_ph, mu_ph]
        dphi_args = [ca.vertcat(*dua_ph), l_ph, dl_ph, s_ph, Q_ph, q_ph, H_ph, g_ph, mu_ph]
        if self.merit_function == 'stat_l1':
            self.f_phi = ca.Function('f_phi', phi_args, [stat_norm + vio])
            self.f_dphi = ca.Function('f_dphi', dphi_args, [dstat_norm + dvio])
        elif self.merit_function == 'stat':
            self.f_phi = ca.Function('f_phi', phi_args, [stat_norm])
            self.f_dphi = ca.Function('f_dphi', dphi_args, [dstat_norm])
        else:
            raise(ValueError(f'Merit function option {self.merit_function} not recognized'))
        self.f_dstat_norm = ca.Function('f_dstat_norm', dphi_args, [dstat_norm])

    def _line_search(self, u, du, l, dl, s, ds, Q, q, G, g, evaluate, merit, d_merit):
        phi = merit(u, l, s, q, G, g)
        dphi = d_merit(u, du, l, dl, s, ds, Q, q, G, g)
        if dphi > 0:
            if self.verbose:
                self.print_method(f'- Line search directional derivative is positive: {dphi}')
        alpha, conv = 1.0, False
        for i in range(self.line_search_iters):
            u_trial = u + alpha*du
            l_trial = l + alpha*dl
            s_trial = s + alpha*ds
            q_trial, G_trial, g_trial, _ = evaluate(u_trial, l_trial, False)
            phi_trial = merit(u_trial, l_trial, s_trial, q_trial, G_trial, g_trial)
            if self.verbose:
                self.print_method(f'- Line search iteration: {i} | merit gap: {phi_trial-(phi + self.beta*alpha*dphi):.4e} | a: {alpha:.4e}')
            if phi_trial <= phi + self.beta*alpha*dphi:
                conv = True
                break
            else:
                alpha *= self.tau
        if not conv:
            if self.verbose:
                self.print_method('- Max iterations reached, line search did not succeed')
            # pdb.set_trace()
        return u_trial, l_trial, phi_trial

    def _watchdog_line_search(self, u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit):
        if self.verbose:
            self.print_method('===================================================')
            self.print_method('Watchdog step acceptance routine')
        qp_solves = 0
        t_hat = 5 # Number of steps where we search for sufficient merit decrease
        phi_log = []

        phi_k = merit(u_k, l_k, s_k, q_k, G_k, g_k)
        phi_log.append(phi_k)
        dphi_k = d_merit(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k)
        if dphi_k > 0:
            if self.verbose:
                self.print_method(f'Time k: Directional derivative is positive: {dphi_k}')

        # Take relaxed (full) step
        u_kp1 = u_k + du_k
        l_kp1 = l_k + dl_k
        s_kp1 = s_k + ds_k
        q_kp1, G_kp1, g_kp1, _ = evaluate(u_kp1, l_kp1, False)
        phi_kp1 = merit(u_kp1, l_kp1, s_kp1, q_kp1, G_kp1, g_kp1)
        phi_log.append(phi_kp1)

        # Check for sufficient decrease w.r.t. time k
        if self.verbose:
            self.print_method(f'Time k+1:')
        if phi_kp1 <= phi_k + self.beta*dphi_k:
            if self.verbose:
                self.print_method(f'Sufficient decrease achieved')
            return u_kp1, l_kp1, qp_solves
        # if self.verbose:
        #     self.print_method(f'Insufficient decrease in merit')

        fail = False
        # Check for sufficient decrease in the next t_hat iterations
        u_t, l_t = u_kp1, l_kp1
        for t in range(t_hat):
            if self.verbose:
                self.print_method(f'Time k+{t+2}:')
            # Compute step at iteration t
            Q_t, q_t, G_t, g_t, _ = evaluate(u_t, l_t, True)
            du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
            qp_solves += 1
            if None in du_t:
                fail = True
                break
            dl_t = l_hat - l_t
            s_t = np.minimum(0, g_t)
            ds_t = g_t + G_t @ du_t - s_t

            # Take full step
            u_tp1 = u_t + du_t
            l_tp1 = l_hat
            s_tp1 = s_t + ds_t
            q_tp1, G_tp1, g_tp1, _ = evaluate(u_tp1, l_tp1, False)
            phi_tp1 = merit(u_tp1, l_tp1, s_tp1, q_tp1, G_tp1, g_tp1)
            phi_log.append(phi_tp1)
            if self.verbose:
                self.print_method(phi_log)
            
            # Check for sufficient decrease w.r.t. base iteration
            if phi_tp1 <= phi_k + self.beta*dphi_k:
                if self.verbose:
                    self.print_method(f'Sufficient decrease achieved')
                return u_tp1, l_tp1, qp_solves
            
            # Update for next iteration
            u_t, l_t = u_tp1, l_tp1
        
        # Insist on merit function decrease
        Q_t, q_t, G_t, g_t, _ = evaluate(u_t, l_t, True)
        du_t, l_hat = self._solve_qp(Q_t, q_t, G_t, g_t)
        qp_solves += 1
        if None in du_t:
            fail = True
        else:
            dl_t = l_hat - l_t
            s_t = np.minimum(0, g_t)
            ds_t = g_t + G_t @ du_t - s_t
            u_tp1, l_tp1, phi_tp1 = self._line_search(u_t, du_t, l_t, dl_t, s_t, ds_t, Q_t, q_t, G_t, g_t, evaluate, merit, d_merit)

        if not fail:
            if phi_tp1 <= phi_k + self.beta*dphi_k:
                return u_tp1, l_tp1, qp_solves
            elif phi_tp1 > phi_k:
                fail = True
            else:
                if self.verbose:
                    self.print_method(f'Insufficient decrease in merit')
                Q_tp1, q_tp1, G_tp1, g_tp1, _ = evaluate(u_tp1, l_tp1, True)
                du_tp1, l_hat = self._solve_qp(Q_tp1, q_tp1, G_tp1, g_tp1)
                if None in du_tp1:
                    if self.verbose:
                        self.print_method(f'QP failed, returning to search along step at time k')
                    u_kp1, l_kp1, phi_kp1 = self._line_search(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit)
                    return u_kp1, l_kp1, qp_solves
                qp_solves += 1
                dl_tp1 = l_hat - l_tp1
                s_tp1 = np.minimum(0, g_tp1)
                ds_tp1 = g_tp1 + G_tp1 @ du_tp1 - s_tp1

                u_tp2, l_tp2, _ = self._line_search(u_tp1, du_tp1, l_tp1, dl_tp1, s_tp1, ds_tp1, Q_tp1, q_tp1, G_tp1, g_tp1, evaluate, merit, d_merit)
                return u_tp2, l_tp2, qp_solves

        if fail:
            if self.verbose:
                self.print_method(f'No decrease in merit, returning to search along step at iteration k')
            u_kp1, l_kp1, phi_kp1 = self._line_search(u_k, du_k, l_k, dl_k, s_k, ds_k, Q_k, q_k, G_k, g_k, evaluate, merit, d_merit)
            return u_kp1, l_kp1, qp_solves

    def _nearestPD(self, A):
        B = (A + A.T)/2
        s, U = np.linalg.eigh(B)
        s[np.where(s < 0)[0]] = 0
        return U @ np.diag(s) @ U.T

    def _update_debug_plot(self, u, x0, up):
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
            raise NotImplementedError('Conversion from local to global pos has not been implemented for debug plot')
        self.ax_xy.set_aspect('equal')
        self.ax_xy.relim()
        self.ax_xy.autoscale_view()
        J = self.f_J(u, x0, up)
        self.ax_xy.set_title(str(J))
        for i in range(self.M):
            self.l_a[i].set_data(np.arange(self.N), u_bar[:,0+int(np.sum(self.num_ua_d[:i]))])
            self.l_s[i].set_data(np.arange(self.N), u_bar[:,1+int(np.sum(self.num_ua_d[:i]))])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == '__main__':
    pass