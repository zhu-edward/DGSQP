#!/usr/bin python3

import numpy as np
import scipy as sp
import casadi as ca

import pathlib
import os
import copy
import shutil
import pdb
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

from typing import List, Dict

from DGSQP.types import VehicleState, VehiclePrediction

from DGSQP.dynamics.dynamics_models import CasadiDecoupledMultiAgentDynamicsModel

from DGSQP.solvers.abstract_solver import AbstractSolver
from DGSQP.solvers.solver_types import IBRParams

class IBR(AbstractSolver):
    def __init__(self, joint_dynamics: CasadiDecoupledMultiAgentDynamicsModel, 
                       costs: List[List[ca.Function]], 
                       agent_constraints: List[ca.Function], 
                       shared_constraints: List[ca.Function],
                       bounds: Dict[str, VehicleState],
                       params=IBRParams()):
        self.joint_dynamics = joint_dynamics
        self.M = self.joint_dynamics.n_a

        self.N = params.N

        self.line_search_iters = params.line_search_iters
        self.ibr_iters = params.ibr_iters

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

        self.num_qa_d = [int(self.joint_dynamics.dynamics_models[a].n_q) for a in range(self.M)]
        self.num_ua_d = [int(self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]
        self.num_ua_el = [int(self.N*self.joint_dynamics.dynamics_models[a].n_u) for a in range(self.M)]

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

        self.n_ca = [[0 for _ in range(self.N+1)] for _ in range(self.M)]
        self.n_cbr = [[0 for _ in range(self.N+1)] for _ in range(self.M)]
        self.n_cs = [0 for _ in range(self.N+1)]
        self.n_c = [0 for _ in range(self.N+1)]
        
        self.state_input_predictions = [VehiclePrediction() for _ in range(self.M)]

        self.n_u = self.joint_dynamics.n_u
        self.n_q = self.joint_dynamics.n_q

        # Convergence tolerance for SQP
        self.p_tol = params.p_tol
        self.d_tol = params.d_tol

        self.alpha = 0.3
        self.use_ps = params.use_ps

        self.debug_plot = params.debug_plot
        self.pause_on_plot = params.pause_on_plot
        self.local_pos = params.local_pos
        if self.debug_plot:
            matplotlib.use('TkAgg')
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            self.ax_a = self.fig.add_subplot(2,2,2)
            self.ax_s = self.fig.add_subplot(2,2,4)
            # self.joint_dynamics.dynamics_models[0].track.remove_phase_out()
            self.joint_dynamics.dynamics_models[0].track.plot_map(self.ax_xy, close_loop=False)
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
            
        self.q_pred = np.zeros((self.N+1, self.n_q))
        self.u_pred = np.zeros((self.N, self.n_u))

        self.q_new = np.zeros((self.N+1, self.n_q))
        self.u_new = np.zeros((self.N+1, self.n_u))

        self.debug = False

        self.u_prev = np.zeros(self.n_u)

        if params.solver_dir:
            self._load_solver()
        else:
            self._build_solver()
        
        self.u_ws = [np.zeros((self.N, self.num_ua_d[a])) for a in range(self.M)]
        if self.use_ps and self.alpha > 0:
            self.l_ws = [np.zeros(np.sum(self.n_c)) if a == 0 else np.zeros(np.sum(self.n_cbr[a])) for a in range(self.M)]
        else:
            self.l_ws = [np.zeros(np.sum(self.n_cbr[a])) for a in range(self.M)]
        self.l_pred = copy.copy(self.l_ws)

        self.initialized = True

    def initialize(self):
        pass

    def set_warm_start(self, u_ws: np.ndarray, l_ws: np.ndarray = None):
        self.u_ws = u_ws
        if l_ws is None:
            if self.use_ps and self.alpha > 0:
                self.l_ws = [np.zeros(np.sum(self.n_c)) if a == 0 else np.zeros(np.sum(self.n_cbr[a])) for a in range(self.M)]
            else:
                self.l_ws = [np.zeros(np.sum(self.n_cbr[a])) for a in range(self.M)]
        else:
            self.l_ws = l_ws

    def step(self, states: List[VehicleState], env_state=None):
        info = self.solve(states)

        self.joint_dynamics.qu2state(states, None, self.u_pred[0])
        self.joint_dynamics.qu2prediction(self.state_input_predictions, self.q_pred, self.u_pred)
        for q in self.state_input_predictions:
            q.t = states[0].t

        self.u_prev = self.u_pred[0]
        
        u_ws = np.vstack((self.u_pred[1:], self.u_pred[-1]))
        u = []
        for a in range(self.M):
            si = int(np.sum(self.num_ua_d[:a]))
            ei = si + int(self.num_ua_d[a])
            u.append(u_ws[:,si:ei].ravel())
        self.set_warm_start(u)

        return info

    def solve(self, states: List[VehicleState]):
        solve_info = {}
        solve_start = datetime.now()

        u_i = []
        for a in range(self.M):
            u_i.append(self.u_ws[a].ravel())
        l_i = copy.copy(self.l_ws)
        x0 = self.joint_dynamics.state2q(states)
        up = copy.copy(self.u_prev)
        u_im1 = copy.copy(u_i)

        if self.verbose:
            J = self.f_J(np.concatenate(u_i), x0, up)
            print(f'ego cost: {J[0]}, tar cost: {J[1]}')

        if self.debug_plot:
            self._update_debug_plot(u_i, x0, up)
            if self.pause_on_plot:
                pdb.set_trace()

        ibr_converged = False
        ibr_it = 0
        iter_sols = []
        while True:
            ibr_it_start = datetime.now()
            iter_sols.append(u_i)
            if self.verbose:
                print('===================================================')
                print(f'IBR iteration: {ibr_it}')

            cond = None

            for a in range(self.M):
            # for a in range(self.M-1, -1, -1):
                # if ibr_it == 0 or not self.use_ps:
                if self.use_ps and a == 0 and self.alpha > 0:
                    # Compute policy gradient
                    Duo_ubr_v = []
                    for b in range(self.M):
                        if b != a:
                            uo = np.concatenate([u_i[c] for c in range(self.M) if c != b])
                            try:
                                Duo_ubr = self.f_Duo_ubr[b](u_i[b], l_i[b], uo, x0, up).toarray()
                                Duo_ubr_v.append(Duo_ubr.ravel(order='F'))

                            except Exception as e:
                                print(e)
                                pdb.set_trace()
                    p = np.concatenate((x0, 
                                        up, 
                                        np.concatenate(u_i), 
                                        np.concatenate(Duo_ubr_v),
                                        np.array([self.alpha])))

                    solver_args = {}
                    solver_args['x0'] = u_i[a]
                    solver_args['lam_g0'] = l_i[a]
                    solver_args['lbx'] = -np.inf*np.ones(self.N*self.num_ua_d[a])
                    solver_args['ubx'] = np.inf*np.ones(self.N*self.num_ua_d[a])
                    solver_args['lbg'] = -np.inf*np.ones(np.sum(self.n_c))
                    solver_args['ubg'] = np.zeros(np.sum(self.n_c))
                    solver_args['p'] = p

                    sol = self.ps_br_solvers[a](**solver_args)
                    if self.verbose:
                        print(self.ps_br_solvers[a].stats()['return_status'])
                    if self.ps_br_solvers[a].stats()['success'] or self.ps_br_solvers[a].stats()['return_status'] == 'Maximum_Iterations_Exceeded':
                        u_i[a] = sol['x'].toarray().squeeze()
                        l_i[a] = sol['lam_g'].toarray().squeeze()
                    else:
                        pdb.set_trace()
                    # G_i[a] = self.f_Dua_Lps[a](np.concatenate(u_i), l_i[a], np.concatenate(u_im1), g, x0, up)
                else:
                    uo = np.concatenate([u_i[b] for b in range(self.M) if b != a])
                    p = np.concatenate((x0, up, uo))

                    solver_args = {}
                    solver_args['x0'] = u_i[a]
                    solver_args['lam_g0'] = l_i[a]
                    solver_args['lbx'] = -np.inf*np.ones(self.N*self.num_ua_d[a])
                    solver_args['ubx'] = np.inf*np.ones(self.N*self.num_ua_d[a])
                    solver_args['lbg'] = -np.inf*np.ones(np.sum(self.n_cbr[a]))
                    solver_args['ubg'] = np.zeros(np.sum(self.n_cbr[a]))
                    solver_args['p'] = p

                    sol = self.br_solvers[a](**solver_args)
                    if self.verbose:
                        print(self.br_solvers[a].stats()['return_status'])
                    if self.br_solvers[a].stats()['success'] or self.br_solvers[a].stats()['return_status'] == 'Maximum_Iterations_Exceeded':
                        u_i[a] = sol['x'].toarray().squeeze()
                        l_i[a] = sol['lam_g'].toarray().squeeze()
                    else:
                        pdb.set_trace()

                if self.debug_plot:
                    u_bar = copy.deepcopy(u_i)
                    if self.use_ps and a == 0 and self.alpha > 0:
                        u_bar[1] += Duo_ubr @ (u_bar[0] - u_im1[0])
                    self._update_debug_plot(u_bar, x0, up)
                    if self.pause_on_plot:
                        pdb.set_trace()

            du = [np.linalg.norm(u_i[a]-u_im1[a]) for a in range(self.M)]
            if self.verbose:
                print('Delta strategy:', du)
            if np.amax(du) < self.p_tol:
                ibr_converged = True
                if self.verbose: print('IBR converged')
                break

            u_im1 = copy.deepcopy(u_i)

            ibr_it_dur = (datetime.now()-ibr_it_start).total_seconds()
            if self.verbose:
                print(f'IBR iteration {ibr_it} time: {ibr_it_dur}')
                # print(f'SQP step size primal: {ps:.4e}, dual: {ds:.4e}')
                # print('SQP iterate: ', u)
                print('===================================================')

            if self.verbose:
                J = self.f_J(np.concatenate(u_i), x0, up)
                print(f'ego cost: {J[0]}, tar cost: {J[1]}')

            ibr_it += 1
            if ibr_it >= self.ibr_iters:
                if self.verbose: print('Max IBR iterations reached')
                break
        
        x_bar = np.array(self.f_state_rollout(np.concatenate(u_i), x0)).squeeze()
        u_bar = []
        for a in range(self.M):
            u_bar.append(u_i[a].reshape((self.N, self.num_ua_d[a])))

        self.q_pred = x_bar
        self.u_pred = np.hstack(u_bar)
        self.l_pred = l_i

        solve_dur = (datetime.now()-solve_start).total_seconds()
        print(f'Solve time: {solve_dur}')
        J = self.f_J(np.concatenate(u_i), x0, up)
        print(f'ego cost: {J[0]}, tar cost: {J[1]}')

        solve_info['time'] = solve_dur
        solve_info['num_iters'] = ibr_it
        solve_info['status'] = ibr_converged
        solve_info['cost'] = J
        solve_info['cond'] = cond
        solve_info['iter_sols'] = iter_sols

        if self.debug_plot:
            plt.ioff()

        return solve_info

    def solve_br(self, state: List[VehicleState], agent_id: int, params: np.ndarray):
        if not self.initialized:
            raise(RuntimeError('NL MPC controller is not initialized, run NL_MPC.initialize() before calling NL_MPC.solve()'))

        x = self.joint_dynamics.state2q(state)

        n_u = self.num_ua_d[agent_id]

        if self.u_ws[agent_id] is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws[agent_id] = np.zeros((self.N, n_u))

        # Construct initial guess for the decision variables and the runtime problem data
        p = np.concatenate((x, self.u_prev, *params))
        
        solver_args = {}
        solver_args['x0'] = self.u_ws[agent_id].ravel()
        solver_args['lbx'] = -np.inf*np.ones(self.N*n_u)
        solver_args['ubx'] = np.inf*np.ones(self.N*n_u)
        solver_args['lbg'] = -np.inf*np.ones(np.sum(self.n_cbr[agent_id]))
        solver_args['ubg'] = np.zeros(np.sum(self.n_cbr[agent_id]))
        solver_args['p'] = p
        # if self.lam_g_ws is not None:
        #     solver_args['lam_g0'] = self.lam_g_ws

        sol = self.br_solvers[agent_id](**solver_args)

        if self.br_solvers[agent_id].stats()['success']:
            # Unpack solution
            u_sol = sol['x'].toarray().squeeze()
            u_joint = []
            i = 0
            for a in range(self.M):
                if a == agent_id:
                    u_joint.append(u_sol)
                else:
                    u_joint.append(params[i])
                    i += 1
            x_pred = np.array(self.f_state_rollout(np.concatenate(u_joint), x)).squeeze()
            u_pred = np.reshape(u_sol, (self.N, n_u))
            # slack_sol = sol['x'][(self.n_q+self.n_u)*self.N:]
            # lam_g_ws = sol['lam_g'].toarray()
            self.x_pred = x_pred
            self.u_pred = u_pred
        else:
            u_joint = []
            i = 0
            for a in range(self.M):
                if a == agent_id:
                    u_joint.append(self.u_pred[-1])
                else:
                    u_joint.append(params[i][-self.num_ua_d[a]:])
                    i += 1
            self.x_pred = np.vstack((x, self.x_pred[2:], self.joint_dynamics.fd(self.x_pred[-1], np.concatenate(u_joint)).toarray().squeeze()))
            self.u_pred = np.vstack((self.u_pred[1:], self.u_pred[-1]))
            # lam_g_ws = np.zeros(np.sum(self.n_ca[agent_id]))
        
        return {'status': self.br_solvers[agent_id].stats()['success'], 
                'stats': self.br_solvers[agent_id].stats(), 
                'sol': sol}

    def _evaluate_br(self, u, l, x0, up):
        u = np.concatenate(u)
        c = [ca.vertcat(*self.f_Cbr[a](u, x0, up)).toarray().squeeze() for a in range(self.M)]
        G = [self.f_Dua_Lbr[a](u, l[a], x0, up).toarray().squeeze() for a in range(self.M)]

        return c, G

    def _evaluate_ps(self, u, l, x0, up):
        u = np.concatenate(u)
        c = ca.vertcat(*self.f_C(u, x0, up)).toarray().squeeze()
        # G = [self.f_Dua_Lps[a](u, l[a], um, g, x0, up).toarray().squeeze() for a in range(self.M)]

        return c

    def _build_solver(self):
        # Build best response OCPs
        # Put optimal control problem in batch form
        x_ph = [ca.MX.sym('x_ph_0', self.n_q)] # Initial state
        # u_0, ..., u_N-1, u_-1
        u_ph = [[ca.MX.sym(f'u{a}_ph_{k}', self.num_ua_d[a]) for k in range(self.N+1)] for a in range(self.M)] # Agent inputs
        ua_ph = [ca.vertcat(*u_ph[a][:-1]) for a in range(self.M)] # [u_0^1, ..., u_{N-1}^1, u_0^2, ..., u_{N-1}^2]
        uk_ph = [ca.vertcat(*[u_ph[a][k] for a in range(self.M)]) for k in range(self.N+1)] # [[u_0^1, u_0^2], ..., [u_{N-1}^1, u_{N-1}^2]]

        for k in range(self.N):
            x_ph.append(self.joint_dynamics.fd(x_ph[k], uk_ph[k]))
        self.f_state_rollout = ca.Function('f_state_rollout', [ca.vertcat(*ua_ph), x_ph[0]], x_ph, self.options)
        
        # Agent cost functions
        J = [ca.DM.zeros(1) for _ in range(self.M)]
        for a in range(self.M):
            for k in range(self.N):
                J[a] += self.costs_sym[a][k](x_ph[k], u_ph[a][k], u_ph[a][k-1])
            J[a] += self.costs_sym[a][-1](x_ph[-1])
        self.f_J = ca.Function('f_J', [ca.vertcat(*ua_ph), x_ph[0], uk_ph[-1]], J, self.options)
        
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
        self.f_C = ca.Function(f'f_C', [ca.vertcat(*ua_ph), x_ph[0], uk_ph[-1]], C, self.options)

        # Best response specific constraint functions
        Cbr = [[[] for _ in range(self.N+1)] for _ in range(self.M)]
        for k in range(self.N+1):
            for a in range(self.M):
                Cbr[a][k] = C[k][self.Cbr_k_idxs[a][k]]
                self.n_cbr[a][k] = Cbr[a][k].shape[0]
        self.f_Cbr = [ca.Function(f'f_C{a}', [ca.vertcat(*ua_ph), x_ph[0], uk_ph[-1]], Cbr[a], self.options) for a in range(self.M)]

        # Symbolic gradients of cost and constraint functions
        Du_J = [ca.jacobian(J[a], ua_ph[a]).T for a in range(self.M)]
        Du_C = ca.jacobian(ca.vertcat(*C), ca.vertcat(*ua_ph))
        Du_Cbr = [ca.jacobian(ca.vertcat(*Cbr[a]), ua_ph[a]) for a in range(self.M)]
        self.f_Du_J = [ca.Function(f'f_Du_J{a}', [ca.vertcat(*ua_ph), x_ph[0], uk_ph[-1]], [Du_J[a]], self.options) for a in range(self.M)]
        self.f_Du_C = ca.Function('f_Du_C', [ca.vertcat(*ua_ph), x_ph[0], uk_ph[-1]], [Du_C], self.options)
        self.f_Du_Cbr = [ca.Function(f'f_Du_Cbr{a}', [ca.vertcat(*ua_ph), x_ph[0], uk_ph[-1]], [Du_Cbr[a]], self.options) for a in range(self.M)]

        # Symbolic gradients of best response Lagrangians
        lbr_ph = [[ca.MX.sym(f'lbr{a}_ph_{k}', self.n_cbr[a][k]) for k in range(self.N+1)] for a in range(self.M)]
        Lbr = [J[a] + ca.dot(ca.vertcat(*lbr_ph[a]), ca.vertcat(*Cbr[a])) for a in range(self.M)]
        Dua_Lbr = [ca.jacobian(Lbr[a], ua_ph[a]).T for a in range(self.M)]
        # Duu_L = [[ca.jacobian(Du_L[a][b], ca.vertcat(*ua_ph)) for b in range(self.M)] for a in range(self.M)]
        self.f_Dua_Lbr = [ca.Function(f'f_Du_Lbr{a}', [ca.vertcat(*ua_ph), ca.vertcat(*lbr_ph[a]), x_ph[0], uk_ph[-1]], [Dua_Lbr[a]], self.options) for a in range(self.M)]
        # self.f_Duu_L = [ca.Function(f'f_Duu_L{a}', [ca.vertcat(*ua_ph), ca.vertcat(*l_ph), x_ph[0], uk_ph[-1]], Duu_L[a], self.options) for a in range(self.M)]

        if self.code_gen and self.jit:
            self.code_gen_opts = dict(jit=True, 
                                jit_name=self.solver_name, 
                                compiler='shell',
                                jit_options=dict(compiler='gcc', flags=['-%s' % self.opt_flag], verbose=self.verbose))
        else:
            self.code_gen_opts = dict(jit=False)

        ipopt_opts = dict(max_iter=200,
                          mu_strategy='adaptive',
                          warm_start_init_point='yes')
        solver_opts = dict(error_on_fail=False, 
                            verbose_init=self.verbose, 
                            ipopt=ipopt_opts,
                            **self.code_gen_opts)

        reg = 1e-3
        Fbr, Dula_Fbr, Duo_Fbr, Duo_ubr = [], [], [], []
        self.f_Fbr, self.f_Dula_Fbr, self.f_Duo_Fbr, self.f_Duo_ubr = [], [], [], []
        self.br_nlps, self.br_solvers = [], []
        for a in range(self.M):
            # The input sequences of all other agents
            uo = [ua_ph[b] for b in range(self.M) if b != a]

            Fbr.append(ca.vertcat(Dua_Lbr[a], ca.vertcat(*lbr_ph[a])*ca.vertcat(*Cbr[a])))
            self.f_Fbr.append(ca.Function(f'f_F{a}', [ua_ph[a], ca.vertcat(*lbr_ph[a]), ca.vertcat(*uo), x_ph[0], uk_ph[-1]], [Fbr[a]], self.options))
            Dula_Fbr.append(ca.jacobian(Fbr[a], ca.vertcat(ua_ph[a], *lbr_ph[a])))
            self.f_Dula_Fbr.append(ca.Function(f'f_Dul{a}_F{a}', [ua_ph[a], ca.vertcat(*lbr_ph[a]), ca.vertcat(*uo), x_ph[0], uk_ph[-1]], [Dula_Fbr[a]], self.options))
            Duo_Fbr.append(ca.jacobian(Fbr[a], ca.vertcat(*uo)))
            self.f_Duo_Fbr.append(ca.Function(f'f_Duo_F{a}', [ua_ph[a], ca.vertcat(*lbr_ph[a]), ca.vertcat(*uo), x_ph[0], uk_ph[-1]], [Duo_Fbr[a]], self.options))
            # Duo_ua.append(-ca.solve(Duala_Fa[a]+reg*ca.DM.eye(Duala_Fa[a].shape[0]), Duo_Fa[a])[:self.N*self.num_ua_d[a],:])
            Duo_ubr.append(-ca.solve(Dula_Fbr[a], Duo_Fbr[a])[:self.N*self.num_ua_d[a],:])
            self.f_Duo_ubr.append(ca.Function(f'f_Duo_u{a}', [ua_ph[a], ca.vertcat(*lbr_ph[a]), ca.vertcat(*uo), x_ph[0], uk_ph[-1]], [Duo_ubr[a]], self.options))

            self.br_nlps.append(dict(x=ua_ph[a], 
                                     p=ca.vertcat(x_ph[0], uk_ph[-1], *uo), 
                                     f=J[a], 
                                     g=ca.vertcat(*Cbr[a])))
            self.br_solvers.append(ca.nlpsol('solver', 'ipopt', self.br_nlps[a], solver_opts))
        
        # Build policy sensitivity best response OCP
        # Nominal input sequences for all agents
        uan_ph = [ca.MX.sym(f'un{a}_ph', ua_ph[a].shape[0]) for a in range(self.M)]
        # Placeholder policy gradient for agent a (ua) w.r.t. other agents (uo)
        Duo_ubr_ph = [ca.MX.sym(f'Duo_u{a}', np.prod(Duo_ubr[a].shape)) for a in range(self.M)]
        alpha = ca.MX.sym('alpha', 1)
        J_ps, C_ps = [], []
        for a in range(self.M):
            ua = []
            for b in range(self.M):
                if b == a:
                    ua.append(ua_ph[b])
                else:
                    if b > a:
                        si = int(np.sum(self.num_ua_el[:a]))
                    else:
                        si = int(np.sum(self.num_ua_el[:a]) - self.num_ua_el[b])
                    ei = int(si + self.num_ua_el[a])
                    dub = ca.reshape(Duo_ubr_ph[b], Duo_ubr[b].shape)[:,si:ei] @ (ua_ph[a] - uan_ph[a])
                    ua.append(uan_ph[b] + alpha*dub)

            # Agent cost functions
            J_ps.append(self.f_J(ca.vertcat(*ua), x_ph[0], uk_ph[-1])[a])
            # Constraint functions for each agent: C(x, u) <= 0
            C_ps.append(self.f_C(ca.vertcat(*ua), x_ph[0], uk_ph[-1]))
        
        ipopt_opts = dict(max_iter=200, 
                          linear_solver='ma27',
                          mu_strategy='adaptive',
                          warm_start_init_point='yes',
                          tol=1e-5)
        solver_opts = dict(error_on_fail=False, 
                            verbose_init=self.verbose, 
                            ipopt=ipopt_opts,
                            **self.code_gen_opts)

        self.ps_br_nlps, self.ps_br_solvers = [], []
        for a in range(self.M):
            self.ps_br_nlps.append(dict(x=ua_ph[a], 
                                        p=ca.vertcat(x_ph[0], uk_ph[-1], *uan_ph, *[Duo_ubr_ph[b] for b in range(self.M) if b != a], alpha), 
                                        f=J_ps[a], 
                                        g=ca.vertcat(*C_ps[a])))
            self.ps_br_solvers.append(ca.nlpsol('solver', 'ipopt', self.ps_br_nlps[a], solver_opts))
        
        # Symbolic gradients of policy sensitive best response Lagrangians
        lps_ph = [[ca.MX.sym(f'lps{a}_ph_{k}', self.n_c[k]) for k in range(self.N+1)] for a in range(self.M)]
        Lps = [J_ps[a] + ca.dot(ca.vertcat(*lps_ph[a]), ca.vertcat(*C_ps[a])) for a in range(self.M)]
        Dua_Lps = [ca.jacobian(Lps[a], ua_ph[a]).T for a in range(self.M)]
        self.f_Dua_Lps = [ca.Function(f'f_Du_Lps{a}', [ca.vertcat(*ua_ph), ca.vertcat(*lps_ph[a]), ca.vertcat(*uan_ph), ca.vertcat(*Duo_ubr_ph), alpha, x_ph[0], uk_ph[-1]], [Dua_Lps[a]], self.options) for a in range(self.M)]

        if self.code_gen and not self.jit:
            generator = ca.CodeGenerator(self.c_file_name)

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
            solver_path = str(pathlib.Path(self.solver_dir, self.so_file_name).expanduser())
        if self.verbose:
            print('- Loading solver from %s' % solver_path)

    def get_prediction(self) -> List[VehiclePrediction]:
        return self.state_input_predictions

    def _update_debug_plot(self, u, x0, up):
        q_nom = np.array(self.f_state_rollout(np.concatenate(u), x0)).squeeze()
        u_nom = []
        for a in range(self.M):
            u_nom.append(u[a].reshape((self.N, self.num_ua_d[a])))
        if not self.local_pos:
            for i in range(self.M):
                self.l_xy[i].set_data(q_nom[:,0+int(np.sum(self.num_qa_d[:i]))], q_nom[:,1+int(np.sum(self.num_qa_d[:i]))])
        else:
            raise NotImplementedError('Conversion from local to global pos has not been implemented for debug plot')
        self.ax_xy.set_aspect('equal')
        J = self.f_J(np.concatenate(u), x0, up)
        self.ax_xy.set_title(str(J))
        for i in range(self.M):
            self.l_a[i].set_data(np.arange(self.N), u_nom[i][:,0])
            self.l_s[i].set_data(np.arange(self.N), u_nom[i][:,1])
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_s.relim()
        self.ax_s.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == '__main__':
    pass
