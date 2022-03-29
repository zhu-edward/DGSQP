#!/usr/bin python3

import numpy as np
import scipy as sp
import casadi as ca

import pathlib
import os
import copy
import shutil
import pdb
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

from typing import List, Dict

from DGSQP.types import VehicleState, VehiclePrediction

from DGSQP.dynamics.dynamics_models import CasadiDecoupledMultiAgentDynamicsModel

from DGSQP.solvers.abstract_solver import AbstractSolver
from DGSQP.solvers.solver_types import ALGAMESParams

class ALGAMES(AbstractSolver):
    def __init__(self, joint_dynamics: CasadiDecoupledMultiAgentDynamicsModel, 
                       costs: List[Dict[str, ca.Function]], 
                       constraints: List[ca.Function], 
                       bounds: Dict[str, VehicleState],
                       params=ALGAMESParams()):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a

        self.N = params.N

        self.outer_iters = params.outer_iters
        self.line_search_iters = params.line_search_iters
        self.newton_iters = params.newton_iters

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
            raise ValueError('Number of agents: %i, but only %i cost functions were provided' % (self.M, len(costs)))
        self.costs_sym = costs

        # The constraints should be a list (of length N+1) of casadi functions such that constraints[i] <= 0
        if len(constraints) != self.N+1:
            raise ValueError('Horizon length: %i, but only %i constraint functions were provide' % (self.N+1, len(constraints)))
        self.constraints_sym = constraints
        # Process box constraints
        self.state_ub, self.input_ub = self.joint_dynamics.state2qu(bounds['ub'])
        self.state_lb, self.input_lb = self.joint_dynamics.state2qu(bounds['lb'])
        self.state_ub_idxs = np.where(self.state_ub < np.inf)[0]
        self.state_lb_idxs = np.where(self.state_lb > -np.inf)[0]
        self.input_ub_idxs = np.where(self.input_ub < np.inf)[0]
        self.input_lb_idxs = np.where(self.input_lb > -np.inf)[0]

        self.n_c = 0
        # for k in range(self.N+1):
        #     self.n_c += self.constraints_sym[k].size1_out(0) # Number of constraints

        self.state_input_predictions = [VehiclePrediction() for _ in range(self.M)]

        self.n_u = self.joint_dynamics.n_u
        self.n_q = self.joint_dynamics.n_q

        self.newton_step_tol = params.newton_step_tol
        # Convergence tolerance for Newton's method
        self.ineq_tol = params.ineq_tol
        self.eq_tol = params.eq_tol
        self.opt_tol = params.opt_tol
        self.rel_tol_req = 5

        # Lagrangian Regularization
        self.rho_init = params.rho
        self.gamma = params.gamma
        self.rho_val = copy.copy(self.rho_init)
        self.rho_max = params.rho_max
        self.lam_max = params.lam_max

        # Jacobian regularization
        self.q_reg_init = params.q_reg
        self.u_reg_init = params.u_reg

        # Line search parameters
        self.beta = params.beta
        self.tau = params.tau
        self.line_search_tol = params.line_search_tol

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

        self.q_ws = None
        self.u_ws = None
        self.l_ws = None
        self.m_ws = None

        self.debug = False

        self.u_prev = np.zeros(self.n_u)

        if params.solver_dir:
            self._load_solver()
        else:
            self._build_solver()

        self.initialized = True

    def initialize(self):
        pass

    def set_warm_start(self, q_ws: np.ndarray, u_ws: np.ndarray, l_ws: np.ndarray = None, m_ws: np.ndarray = None):
        if q_ws.shape[0] != self.N+1 or q_ws.shape[1] != self.n_q:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (q_ws.shape[0],q_ws.shape[1],self.N+1,self.n_q)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.n_u:
            raise(RuntimeError('Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (u_ws.shape[0],u_ws.shape[1],self.N,self.n_u)))
        self.q_ws = q_ws
        self.u_ws = u_ws

        self.l_ws = l_ws
        self.m_ws = m_ws

    def step(self, states: List[VehicleState], env_state=None):
        info = self.solve(states)

        self.joint_dynamics.qu2state(states, None, self.u_pred[0])
        self.joint_dynamics.qu2prediction(self.state_input_predictions, self.q_pred, self.u_pred)
        for q in self.state_input_predictions:
            q.t = states[0].t

        self.u_prev = self.u_pred[0]

        q_ws = np.vstack((self.q_pred[1:], self.joint_dynamics.fd(self.q_pred[-1], self.u_pred[-1]).toarray().squeeze()))
        u_ws = np.vstack((self.u_pred[1:], self.u_pred[-1]))
        # self.set_warm_start(q_ws, u_ws, lam_bar.toarray(), mu_bar.toarray())
        self.set_warm_start(q_ws, u_ws)

        return info

    def solve(self, states: List[VehicleState]):
        solve_info = {}
        solve_start = datetime.now()
        self.u_prev = np.zeros(self.n_u)

        if self.q_ws is None or self.u_ws is None:
            # Rollout trajectory using input sequence from last solve
            q_bar = np.zeros((self.N+1, self.n_q))
            q_bar[0] = self.joint_dynamics.state2q(states)
            u_bar = np.vstack((self.u_pred[1:], self.u_pred[-1].reshape((1,-1)), self.u_prev.reshape((1,-1))))
            for k in range(self.N):
                # Update dynamics
                q_bar[k+1] = self.joint_dynamics.fd(q_bar[k], u_bar[k]).toarray().squeeze()
        else:
            q_bar = copy.copy(self.q_ws)
            u_bar = np.vstack((copy.copy(self.u_ws), self.u_prev.reshape((1,-1))))
        
        q_bar = ca.DM(q_bar.T)
        u_bar = ca.DM(u_bar.T)
        
        lam_bar = ca.DM.zeros(self.n_c)
        mu_bar = ca.DM.zeros((self.n_q*self.N, self.M))
        
        init = dict(q=copy.copy(q_bar), 
                    u=copy.copy(u_bar), 
                    l=copy.copy(lam_bar), 
                    m=copy.copy(mu_bar))

        if self.debug_plot:
            self._update_debug_plot(q_bar, u_bar)
            if self.pause_on_plot:
                pdb.set_trace()

        q_reg = copy.copy(self.q_reg_init)
        u_reg = copy.copy(self.u_reg_init)
        self.rho_val = copy.copy(self.rho_init)

        # Do ALGAMES
        converged = False
        rel_tol_its = 0
        iter_data = []
        print('ALGAMES')
        for i in range(self.outer_iters):
            it_start = datetime.now()
            if self.verbose:
                print('===================================================')
                print(f'ALGAMES iteration: {i}')
            
            u_im1 = copy.copy(u_bar)
            l_im1 = copy.copy(lam_bar)
            m_im1 = copy.copy(mu_bar)

            # Compute constraint violation for initial guess and construct inital regularization matrix
            C_bar = self.f_C(*ca.horzsplit(q_bar, 1), *ca.horzsplit(u_bar, 1))
            rho_bar = ca.DM([0 if c < 0 and l == 0 else self.rho_val for (c, l) in zip(ca.vertsplit(C_bar), ca.vertsplit(lam_bar))])
            # rho_bar = ca.DM([0 if c < -1e-7 and l < 1e-7 else self.rho_val for (c, l) in zip(ca.vertsplit(C_bar), ca.vertsplit(lam_bar))])

            # Newton's method w/ backtracking line search
            newton_converged = False
            for j in range(self.newton_iters):
                # Scheduled increase of regularization
                q_reg_it = q_reg*(j+1)**4 
                u_reg_it = u_reg*(j+1)**4
                # Compute search direction
                dq, du, dm, Gy = self.f_dy(*ca.horzsplit(q_bar, 1), 
                            *ca.horzsplit(u_bar, 1),
                            *ca.horzsplit(mu_bar, 1),
                            lam_bar,
                            rho_bar,
                            q_reg_it,
                            u_reg_it)
                if ca.norm_inf(Gy) < self.opt_tol:
                    if self.verbose:
                        print(f' - Newton iteration: {j} | G norm: {np.linalg.norm(Gy, ord=np.inf):.4e} | converged: Gradient of Lagrangian within specified tolerance')
                    newton_converged = True
                    newton_status = 'stat_size'
                    Gy_bar = ca.DM(Gy)
                    break

                norm_Gy = np.linalg.norm(Gy, ord=1)/Gy.size1()

                # Do line search
                line_search_converged = False
                alpha = 1.0
                q_tmp = ca.DM(q_bar); u_tmp = ca.DM(u_bar); mu_tmp = ca.DM(mu_bar)
                for k in range(self.line_search_iters):
                    q_trial = q_tmp + ca.horzcat(ca.DM.zeros((self.n_q, 1)), alpha*dq)
                    u_trial = u_tmp + ca.horzcat(alpha*du, ca.DM.zeros((self.n_u, 1)))
                    mu_trial = mu_tmp + alpha*dm
                    Gy_trial = self.f_G_reg(*ca.horzsplit(q_trial, 1), 
                            *ca.horzsplit(u_trial, 1),
                            *ca.horzsplit(mu_trial, 1),
                            lam_bar,
                            rho_bar,
                            q_reg_it,
                            u_reg_it,
                            *ca.horzsplit(q_bar, 1), 
                            *ca.horzsplit(u_bar, 1))
                    norm_Gy_trial = np.linalg.norm(Gy_trial, ord=1)/Gy_trial.size1()
                    norm_Gy_thresh = (1-alpha*self.beta)*norm_Gy
                    if self.verbose:
                        print(f'   - Line search iteration: {k} | LS G norm: {norm_Gy_trial:.4e} | G norm: {norm_Gy_thresh:.4e} | a: {alpha:.4e}')
                    # if norm_Gy_trial-norm_Gy_thresh <= 1e-3:
                    if norm_Gy_trial <= norm_Gy_thresh:
                        line_search_converged = True
                        break
                    else:
                        alpha *= self.tau
                q_bar = ca.DM(q_trial); u_bar = ca.DM(u_trial); mu_bar = ca.DM(mu_trial); Gy_bar = ca.DM(Gy_trial)
                if not line_search_converged:
                    if self.verbose:
                        print('   - Max line search iterations reached, did not converge')
                        print(f' - Newton iteration: {j} | Line search did not converge')
                    newton_converged = False
                    newton_status = 'ls_fail'
                    break

                # Compute average step size
                d = 0
                for k in range(self.N):
                    d += (np.linalg.norm(dq[:,k], ord=1) + np.linalg.norm(du[:,k], ord=1))
                d *= (alpha/((self.n_q + self.n_u)*self.N))

                if self.debug:
                    pdb.set_trace()

                # Check for convergence
                if d < self.newton_step_tol:
                    if self.verbose:
                        print(f' - Newton iteration: {j} | converged: Average step size within specified tolerance')
                    newton_converged = True
                    newton_status = 'step_size'
                    break
                
                if self.verbose:
                    print(f' - Newton iteration: {j} | G norm: {np.linalg.norm(Gy_bar, ord=np.inf):.4e} | step size: {d:.4e} | reg: {u_reg_it:.4e}')
            newton_solves = j + 1
            if newton_solves == self.newton_iters:
                newton_status = 'max_it'
                if self.verbose:
                    print(f' - Newton iteration: {j} | Max Newton iterations reached, did not converge')

            # Compute constraint violation
            ineq_val, eq_val = self.f_CD(*ca.horzsplit(q_bar, 1), *ca.horzsplit(u_bar, 1))
            max_ineq_vio = np.linalg.norm(ca.fmax(ineq_val, ca.DM.zeros(self.n_c)), ord=np.inf)
            max_eq_vio = np.linalg.norm(eq_val, ord=np.inf)
            max_opt_vio = np.linalg.norm(self.f_opt(*ca.horzsplit(q_bar, 1), 
                            *ca.horzsplit(u_bar, 1),
                            *ca.horzsplit(mu_bar, 1),
                            lam_bar), ord=np.inf)
            comp = float(ca.dot(lam_bar, ineq_val))
            cond = {'p_feas': max(max_ineq_vio, max_eq_vio), 'd_feas': 0, 'comp': comp, 'stat': max_opt_vio}

            if self.verbose:
                print(f'ALGAMES iteration: {i} | ineq vio: {max_ineq_vio:.4e} | eq vio: {max_eq_vio:.4e} | comp vio: {comp:.4e} | opt vio: {max_opt_vio:.4e}')
            
            if max_ineq_vio < self.ineq_tol \
                and max_eq_vio < self.eq_tol \
                and comp < self.opt_tol \
                and max_opt_vio < self.opt_tol:
                if self.verbose:
                    print('ALGAMES iterations converged within specified tolerances')
                    print('===================================================')
                it_dur = (datetime.now()-it_start).total_seconds()
                iter_data.append(dict(cond=cond,
                            newton_solves=newton_solves,
                            newton_converged=newton_converged,
                            newton_status=newton_status,
                            it_time=it_dur,
                            u_sol=copy.copy(u_bar), 
                            l_sol=copy.copy(lam_bar), 
                            m_sol=copy.copy(mu_bar)))
                msg = 'conv_abs_tol'
                converged = True
                self.q_pred = copy.copy(q_bar.toarray().T)
                self.u_pred = copy.copy(u_bar[:,:-1].toarray().T)
                if self.debug_plot:
                    self._update_debug_plot(q_bar, u_bar)
                    if self.pause_on_plot:
                        pdb.set_trace()
                break
            
            if np.linalg.norm(u_bar[:,:-1].toarray().ravel()-u_im1[:,:-1].toarray().ravel()) < self.opt_tol/2 \
                    and np.linalg.norm(lam_bar.toarray()-l_im1.toarray()) < self.opt_tol/2 \
                    and np.linalg.norm(mu_bar.toarray().ravel()-m_im1.toarray().ravel()) < self.opt_tol/2:
                rel_tol_its += 1
                if rel_tol_its >= self.rel_tol_req and max_ineq_vio < self.ineq_tol and max_eq_vio < self.eq_tol:
                    it_dur = (datetime.now()-it_start).total_seconds()
                    iter_data.append(dict(cond=cond,
                                newton_solves=newton_solves,
                                newton_converged=newton_converged,
                                newton_status=newton_status,
                                it_time=it_dur,
                                u_sol=copy.copy(u_bar), 
                                l_sol=copy.copy(lam_bar), 
                                m_sol=copy.copy(mu_bar)))
                    converged = True
                    msg = 'conv_rel_tol'
                    if self.verbose: print('ALGAMES iterations converged via relative tolerance')
                    break
            else:
                rel_tol_its = 0

            if max_opt_vio > 1e5:
                it_dur = (datetime.now()-it_start).total_seconds()
                iter_data.append(dict(cond=cond,
                                newton_solves=newton_solves,
                                newton_converged=newton_converged,
                                newton_status=newton_status,
                                it_time=it_dur,
                                u_sol=copy.copy(u_bar), 
                                l_sol=copy.copy(lam_bar), 
                                m_sol=copy.copy(mu_bar)))
                if self.verbose:
                    print('ALGAMES diverged')
                    print('===================================================')
                msg = 'diverged'
                converged = False
                break

            # Do dual ascent
            for k in range(self.n_c):
                lam_bar[k] = min(max(0, lam_bar[k]+rho_bar[k]*ineq_val[k]), self.lam_max) # Update ineq multipliers
            # Scheduled increase of rho
            self.rho_val = min(self.rho_max, self.gamma*self.rho_val)

            it_dur = (datetime.now()-it_start).total_seconds()
            if self.verbose:
                print(f'ALGAMES iteration time: {it_dur}')

            iter_data.append(dict(cond=cond,
                                newton_solves=newton_solves,
                                newton_converged=newton_converged,
                                newton_status=newton_status,
                                it_time=it_dur,
                                u_sol=copy.copy(u_bar), 
                                l_sol=copy.copy(lam_bar), 
                                m_sol=copy.copy(mu_bar)))

            if self.debug:
                pdb.set_trace()
            if self.debug_plot:
                self._update_debug_plot(q_bar, u_bar)
                if self.pause_on_plot:
                    pdb.set_trace()

        if not converged and i == self.outer_iters-1:
            if self.verbose:
                # print('Max ALGAMES iterations reached, did not converge, using best solution from iter %i' % self.best_iter)
                print('Max ALGAMES iterations reached, did not converge')
                print('===================================================')
            msg = 'max_it'
            self.q_pred = copy.copy(q_bar.toarray().T)
            self.u_pred = copy.copy(u_bar[:,:-1].toarray().T)

        solve_dur = (datetime.now()-solve_start).total_seconds()
        print(f'Solve status: {msg}')
        print(f'Solve iters: {i+1}')
        print(f'Solve time: {solve_dur}')
        J = self.f_J(*ca.horzsplit(q_bar, 1), *ca.horzsplit(u_bar, 1))
        print(f'ego cost: {J[0]}, tar cost: {J[1]}')

        solve_info['time'] = solve_dur
        solve_info['num_iters'] = i+1
        solve_info['status'] = converged
        solve_info['cost'] = J
        solve_info['cond'] = cond
        solve_info['iter_data'] = iter_data
        solve_info['msg'] = msg
        solve_info['init'] = init

        if self.debug_plot:
            plt.ioff()

        return solve_info

    def _build_solver(self):
        # =================================
        # Create Lagrangian
        # =================================
        # Placeholder symbolic variables
        q_ph = [ca.MX.sym('q_ph_%i' % k, self.n_q) for k in range(self.N+1)] # Joint state
        ui_ph = [[ca.MX.sym('u_%i_ph_%i' % (i, k), self.joint_dynamics.dynamics_models[i].n_u) for k in range(self.N+1)] for i in range(self.M)] # Agent input
        u_ph = [ca.vertcat(*[ui_ph[i][k] for i in range(self.M)]) for k in range(self.N+1)]
        m_ph = [ca.MX.sym('m_ph_%i' % i, self.n_q*self.N) for i in range(self.M)] # Kinodynamic eq constraint multipliers
        q_ref_ph = [ca.MX.sym('q_ref_ph_%i' % k, self.n_q) for k in range(self.N+1)] # Joint state
        ui_ref_ph = [[ca.MX.sym('u_%i_ref_ph_%i' % (i, k), self.joint_dynamics.dynamics_models[i].n_u) for k in range(self.N+1)] for i in range(self.M)] # Agent input
        u_ref_ph = [ca.vertcat(*[ui_ref_ph[i][k] for i in range(self.M)]) for k in range(self.N+1)]

        # Cost over the horizon
        J = [ca.DM.zeros(1) for i in range(self.M)]
        for i in range(self.M):
            for k in range(self.N):
                J[i] += self.costs_sym[i][k](q_ph[k], ui_ph[i][k], ui_ph[i][k-1])
            J[i] += self.costs_sym[i][-1](q_ph[-1])
        self.f_J = ca.Function('J', q_ph + u_ph, J)
        
        Dq_J = [ca.jacobian(J[a], ca.vertcat(*q_ph)).T for a in range(self.M)]
        Du_J = [ca.jacobian(J[a], ca.vertcat(*ui_ph[a])).T for a in range(self.M)]
        self.f_Dq_J = ca.Function(f'f_Dq_J', q_ph + u_ph, Dq_J)
        self.f_Du_J = ca.Function(f'f_Du_J', q_ph + u_ph, Du_J)

        # Residual of kinodynamic constraints
        D = []
        for k in range(self.N):
            D.append(q_ph[k+1] - self.joint_dynamics.fd(q_ph[k], u_ph[k]))
            # D.append(self.joint_dynamics.fd(q_ph[k], u_ph[k]) - q_ph[k+1])
        D = ca.vertcat(*D)
        self.f_D = ca.Function('D', q_ph + u_ph, [D])

        Dq_D = [ca.jacobian(D, ca.vertcat(*q_ph))]
        Du_D = [ca.jacobian(D, ca.vertcat(*ui_ph[a])) for a in range(self.M)]
        self.f_Dq_D = ca.Function('f_Dq_D', q_ph + u_ph, Dq_D)
        self.f_Du_D = ca.Function('f_Du_D', q_ph + u_ph, Du_D)

        # Residual of inequality constraints
        C = []
        for k in range(self.N):
            if self.constraints_sym[k] is not None:
                C.append(self.constraints_sym[k](q_ph[k], u_ph[k], u_ph[k-1]))
            # Add box constraints
            if len(self.input_ub_idxs) > 0:
                C.append(u_ph[k][self.input_ub_idxs] - self.input_ub[self.input_ub_idxs])
            if len(self.input_lb_idxs) > 0:
                C.append(self.input_lb[self.input_lb_idxs] - u_ph[k][self.input_lb_idxs])
            if len(self.state_ub_idxs) > 0:
                C.append(q_ph[k][self.state_ub_idxs] - self.state_ub[self.state_ub_idxs])
            if len(self.state_lb_idxs) > 0:
                C.append(self.state_lb[self.state_lb_idxs] - q_ph[k][self.state_lb_idxs])
        if self.constraints_sym[-1] is not None:
            C.append(self.constraints_sym[-1](q_ph[-1]))
        # Add box constraints
        if len(self.state_ub_idxs) > 0:
            C.append(q_ph[-1][self.state_ub_idxs] - self.state_ub[self.state_ub_idxs])
        if len(self.state_lb_idxs) > 0:
            C.append(self.state_lb[self.state_lb_idxs] - q_ph[-1][self.state_lb_idxs])
        C = ca.vertcat(*C)
        self.n_c = C.shape[0]
        self.f_C = ca.Function('C', q_ph + u_ph, [C])
        self.f_CD = ca.Function('CD',  q_ph + u_ph, [C, D])

        Dq_C = [ca.jacobian(C, ca.vertcat(*q_ph))]
        Du_C = [ca.jacobian(C, ca.vertcat(*ui_ph[a])) for a in range(self.M)]
        self.f_Dq_C = ca.Function('f_Dq_C', q_ph + u_ph, Dq_C)
        self.f_Du_C = ca.Function('f_Du_C', q_ph + u_ph, Du_C)

        l_ph = ca.MX.sym('l_ph', self.n_c) # Ineq constraint multipliers
        jac_reg_q_ph = ca.MX.sym('jac_reg_q_ph', 1)
        jac_reg_u_ph = ca.MX.sym('jac_reg_u_ph', 1)
        reg_ph = ca.MX.sym('reg_ph', self.n_c)
        
        Lr = []
        for i in range(self.M):
            Lr.append(J[i] + ca.dot(m_ph[i], D) + ca.dot(l_ph, C))
        opt = []
        for i in range(self.M):
            opt_qi, opt_ui = [], []
            for k in range(self.N):
                opt_qi.append(ca.jacobian(Lr[i], q_ph[k+1]).T)
                opt_ui.append(ca.jacobian(Lr[i], ui_ph[i][k]).T)
            # pdb.set_trace()
            opt.append(ca.vertcat(*opt_qi, *opt_ui))
        opt = ca.vertcat(*opt)
        self.f_opt = ca.Function('opt', q_ph + u_ph + m_ph + [l_ph], [opt])


        L = []
        for i in range(self.M):
            L.append(J[i] + ca.dot(m_ph[i], D) + ca.dot(l_ph, C) + ca.bilin(ca.diag(reg_ph), C, C)/2)

        # Gradient of agent Lagrangian w.r.t. joint state and agent input
        G = []
        for i in range(self.M):
            G_qi, G_ui = [], []
            for k in range(self.N):
                G_qi.append(ca.jacobian(L[i], q_ph[k+1]).T)
                G_ui.append(ca.jacobian(L[i], ui_ph[i][k]).T)
            # pdb.set_trace()
            G.append(ca.vertcat(*G_qi, *G_ui))
        G = ca.vertcat(*G, D)
        self.f_G = ca.Function('G', q_ph + u_ph + m_ph + [l_ph, reg_ph], [G])
        
        # Regularized gradient
        G_reg = []
        for i in range(self.M):
            G_qi, G_ui = [], []
            for k in range(self.N):
                G_qi.append(ca.jacobian(L[i], q_ph[k+1]).T + jac_reg_q_ph*(q_ph[k+1]-q_ref_ph[k+1]))
                G_ui.append(ca.jacobian(L[i], ui_ph[i][k]).T + jac_reg_u_ph*(ui_ph[i][k]-ui_ref_ph[i][k]))
            G_reg.append(ca.vertcat(*G_qi, *G_ui))
        G_reg = ca.vertcat(*G_reg, D)
        self.f_G_reg = ca.Function('G_reg', q_ph + u_ph + m_ph + [l_ph, reg_ph, jac_reg_q_ph, jac_reg_u_ph] + q_ref_ph + u_ref_ph, [G_reg])

        # Gradient of G w.r.t. state trajectory (not including initial state), input sequence, and eq constraint multipliers
        y = ca.vertcat(*q_ph[1:], *u_ph[:-1], *m_ph)
        H = ca.jacobian(G, y)
        reg = ca.vertcat(jac_reg_q_ph*ca.DM.ones(self.n_q*self.N), jac_reg_u_ph*ca.DM.ones(self.n_u*self.N), ca.DM.zeros(self.n_q*self.N*self.M))
        H_reg = H + ca.diag(reg)
        self.f_H = ca.Function('H', q_ph + u_ph + m_ph + [l_ph, reg_ph, jac_reg_q_ph, jac_reg_u_ph], [H_reg])
        
        # Search direction
        dy = -ca.solve(H_reg, G, 'lapacklu')
        # dy = -ca.solve(H_reg, G)

        dq = ca.reshape(dy[:self.n_q*self.N], (self.n_q, self.N))
        du = ca.reshape(dy[self.n_q*self.N:self.n_q*self.N+self.n_u*self.N], (self.n_u, self.N))
        dm = ca.reshape(dy[self.n_q*self.N+self.n_u*self.N:], (self.n_q*self.N, self.M))
        self.f_dy = ca.Function('dy', q_ph + u_ph + m_ph + [l_ph, reg_ph, jac_reg_q_ph, jac_reg_u_ph], [dq, du, dm, G])

        if self.code_gen and not self.jit:
            generator = ca.CodeGenerator(self.c_file_name)
            generator.add(self.f_dy)
            generator.add(self.f_J)
            generator.add(self.f_G)
            generator.add(self.f_C)
            generator.add(self.f_CD)

            # Set up paths
            cur_dir = pathlib.Path.cwd()
            gen_path = cur_dir.joinpath(self.solver_name)
            c_path = gen_path.joinpath(self.c_file_name)
            if gen_path.exists():
                shutil.rmtree(gen_path)
            gen_path.mkdir(parents=True)

            os.chdir(gen_path)
            if self.verbose:
                print('- Generating C code for solver %s at %s' % (self.solver_name, str(gen_path)))
            generator.generate()
            pdb.set_trace()
            # Compile into shared object
            so_path = gen_path.joinpath(self.so_file_name)
            if self.verbose:
                print('- Compiling shared object %s from %s' % (so_path, c_path))
                print('- Executing "gcc -fPIC -shared -%s %s -o %s"' % (self.opt_flag, c_path, so_path))
            os.system('gcc -fPIC -shared -%s %s -o %s' % (self.opt_flag, c_path, so_path))

            # Swtich back to working directory
            os.chdir(cur_dir)
            install_dir = self.install()

            # Load solver
            self._load_solver(install_dir.joinpath(self.so_file_name))

    def _load_solver(self, solver_path=None):
        if solver_path is None:
            solver_path = pathlib.Path(self.solver_dir, self.so_file_name).expanduser()
        if self.verbose:
            print('- Loading solver from %s' % str(solver_path))
        self.f_dy = ca.external('dy', str(solver_path))
        self.f_G = ca.external('G', str(solver_path))
        self.f_C = ca.external('C', str(solver_path))
        self.f_J = ca.external('J', str(solver_path))
        self.f_CD = ca.external('CD', str(solver_path))

    def get_prediction(self) -> List[VehiclePrediction]:
        return self.state_input_predictions
    
    def _update_debug_plot(self, q_nom, u_nom):
        if not self.local_pos:
            self.l1_xy.set_data(q_nom.toarray()[0,:], q_nom.toarray()[1,:])
            self.l2_xy.set_data(q_nom.toarray()[0+self.joint_dynamics.dynamics_models[0].n_q,:], q_nom.toarray()[1+self.joint_dynamics.dynamics_models[0].n_q,:])
        else:
            raise NotImplementedError('Conversion from local to global pos has not been implemented for debug plot')
        self.ax_xy.set_aspect('equal')
        J = self.f_J(*ca.horzsplit(q_nom, 1), *ca.horzsplit(u_nom, 1))
        self.ax_xy.set_title(f'ego cost: {J[0]}, tar cost: {J[1]}')
        self.l1_a.set_data(np.arange(self.N), u_nom.toarray()[0,:-1])
        self.l1_s.set_data(np.arange(self.N), u_nom.toarray()[1,:-1])
        self.l2_a.set_data(np.arange(self.N), u_nom.toarray()[2,:-1])
        self.l2_s.set_data(np.arange(self.N), u_nom.toarray()[3,:-1])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == '__main__':
    pass
