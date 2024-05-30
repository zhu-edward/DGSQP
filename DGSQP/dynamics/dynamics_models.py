#!/usr/bin python3

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import casadi as ca

from abc import abstractmethod
import array
from typing import Tuple, List
import copy

from DGSQP.types import VehicleState, VehicleActuation, VehiclePrediction
from DGSQP.dynamics.model_types import *
from DGSQP.dynamics.abstract_model import AbstractModel
from DGSQP.tracks.track_lib import get_track
from DGSQP.tracks.radius_arclength_track import RadiusArclengthTrack
from DGSQP.tracks.casadi_bspline_track import CasadiBSplineTrack

class CasadiDynamicsModel(AbstractModel):
    '''
    Base class for dynamics models that use casadi for their models.
    Implements common functions for linearizing models, integrating models, etc...
    '''

    # whether or not the dynamics depend on curvature
    # curvature is not a state variable as it must be calculated from the track definition
    # so any such model has a third input that is needed for calculations!
    curvature_model = False

    def __init__(self, t0: float, model_config: DynamicsConfig, track=None):
        super().__init__(model_config)

        self.t0 = t0
        # Try to Load track by name first, then by track object passed into model init
        if model_config.track_name is not None:
            self.track = get_track(model_config.track_name)
        else:
            self.track = track

        self.dt = model_config.dt
        self.M = model_config.M # RK4 integration steps
        self.h = self.dt/self.M # RK4 integration time intervals

    @abstractmethod
    def state2qu(self):
        pass

    @abstractmethod
    def state2q(self):
        pass

    @abstractmethod
    def input2u(self):
        pass

    @abstractmethod
    def q2state(self):
        pass

    def precompute_model(self):
        '''
        wraps up model initialization
        require the following fields to be initialized:
        self.sym_q:  ca.SX with elements of state vector q
        self.sym_u:  ca.SX with elements of control vector u
        self.sym_dq: ca.SX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''

        dyn_inputs = [self.sym_q, self.sym_u]
        if type(self.dt) is ca.SX or type(self.dt) is ca.MX:
            dyn_inputs += [self.dt]

        # Continuous time dynamics function
        self.fc = ca.Function('fc', dyn_inputs, [self.sym_dq], self.options('fc'))

        # First derivatives
        self.sym_Ac = ca.jacobian(self.sym_dq, self.sym_q)
        self.sym_Bc = ca.jacobian(self.sym_dq, self.sym_u)
        self.sym_Cc = self.sym_dq

        self.fA = ca.Function('fA', dyn_inputs, [self.sym_Ac], self.options('fA'))
        self.fB = ca.Function('fB', dyn_inputs, [self.sym_Bc], self.options('fB'))
        self.fC = ca.Function('fC', dyn_inputs, [self.sym_Cc], self.options('fC'))

        # Discretization
        discretization_method = self.model_config.discretization_method
        if discretization_method == 'euler':
            sym_q_kp1 = self.sym_q + self.dt * self.fc(*dyn_inputs)
        elif discretization_method == 'rk4':
            sym_q_kp1 = self.rk4(*dyn_inputs, self.fc, self.M, self.h)
        elif discretization_method == 'rk3':
            sym_q_kp1 = self.rk3(*dyn_inputs, self.fc, self.M, self.h)
        elif discretization_method == 'rk2':
            sym_q_kp1 = self.rk2(*dyn_inputs, self.fc, self.M, self.h)
        elif discretization_method == 'idas':
            prob = {'x': self.sym_q, 'p': self.sym_u, 'ode': self.fc(self.sym_q, self.sym_u)}
            setup = {'t0': 0, 'tf': self.dt}
            self.integrator = ca.integrator('int', 'idas', prob, setup)
            if isinstance(self.sym_q, ca.SX):
                self.sym_q = ca.MX.sym('q', self.n_q)
            if isinstance(self.sym_u, ca.SX):
                self.sym_u = ca.MX.sym('u', self.n_u)
            sym_q_kp1 = self.integrator.call([self.sym_q, self.sym_u, 0, 0, 0, 0])[0]
            dyn_inputs[0] = self.sym_q
            dyn_inputs[1] = self.sym_u
        else:
            raise ValueError('Discretization method of %s not recognized' % discretization_method)

        if self.model_config.noise:
            if self.model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to dynamics model')
            noise_cov = np.array(self.model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

            # Symbolic variables for additive process noise components for each state
            self.n_m = self.n_q
            self.sym_m = ca.SX.sym('m', self.n_m)
            sym_q_kp1 += ca.mtimes(la.sqrtm(self.noise_cov), self.sym_m)
            dyn_inputs += [self.sym_m]

        # Discrete time dynamics function
        self.fd = ca.Function('fd', dyn_inputs, [sym_q_kp1], self.options('fd'))

        # First derivatives
        self.sym_Ad = ca.jacobian(sym_q_kp1, self.sym_q)
        self.sym_Bd = ca.jacobian(sym_q_kp1, self.sym_u)
        self.sym_Cd = sym_q_kp1

        self.fAd = ca.Function('fAd', dyn_inputs, [self.sym_Ad], self.options('fAd'))
        self.fBd = ca.Function('fBd', dyn_inputs, [self.sym_Bd], self.options('fBd'))
        self.fCd = ca.Function('fCd', dyn_inputs, [self.sym_Cd], self.options('fCd'))

        # Second derivatives
        if self.model_config.compute_hessians:
            self.sym_Ed = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_q), self.sym_q) for i in range(self.n_q)]
            self.sym_Fd = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_u), self.sym_u) for i in range(self.n_q)]
            self.sym_Gd = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_u), self.sym_q) for i in range(self.n_q)]

            self.fEd = ca.Function('fEd', dyn_inputs, self.sym_Ed, self.options('fEd'))
            self.fFd = ca.Function('fFd', dyn_inputs, self.sym_Fd, self.options('fFd'))
            self.fGd = ca.Function('fGd', dyn_inputs, self.sym_Gd, self.options('fGd'))
        
        if self.model_config.noise:
            self.sym_Md = ca.jacobian(sym_q_kp1, self.sym_m)
            self.fMd = ca.Function('fMd', dyn_inputs, [self.sym_Md], self.options('fMd'))

        # Build shared object if not doing just-in-time compilation
        if self.code_gen and not self.jit:
            so_fns = [self.fc, self.fA, self.fB, self.fC, self.fd, self.fAd, self.fBd, self.fCd]
            if self.model_config.compute_hessians:
                so_fns += [self.fEd, self.fFd, self.fGd]
            if self.model_config.noise:
                so_fns += [self.fMd]
            self.install_dir = self.build_shared_object(so_fns)

        return

    def step(self, vehicle_state: VehicleState):
        '''
        steps noise-free model forward one time step (self.dt) using numerical integration
        '''
        q, u = self.state2qu(vehicle_state)
        t = vehicle_state.t - self.t0
        tf = t + self.dt

        # q_n = self.rk4(q, u, self.fc, self.M, self.h).toarray().squeeze()
        sol = solve_ivp(lambda t, z: np.array(self.fc(z, u)).squeeze(), (0, self.dt), q, t_eval=[self.dt])
        q_n = sol.y.squeeze()

        a_x, a_y, a_z = self.f_a(q_n, u)
        a_phi, a_the, a_psi = self.f_ang_a(q_n, u)

        self.qu2state(vehicle_state, q_n, u)
        vehicle_state.t = tf + self.t0
        vehicle_state.a.a_long, vehicle_state.a.a_tran, vehicle_state.a.a_n = float(a_x), float(a_y), float(a_z)
        vehicle_state.aa.a_phi, vehicle_state.aa.a_theta, vehicle_state.aa.a_psi = float(a_phi), float(a_the), float(a_psi)

        if self.track is not None:
            if self.curvature_model:
                self.track.local_to_global_typed(vehicle_state)
            else:
                self.track.global_to_local_typed(vehicle_state)
            return

    def rk4(self, x, u, f, M, h):
        '''
        Discrete nonlinear dynamics (RK4 approx.)
        '''
        x_p = x
        for _ in range(M):
            a1 = f(x_p, u)
            a2 = f(x_p + (h / 2) * a1, u)
            a3 = f(x_p + (h / 2) * a2, u)
            a4 = f(x_p + h * a3, u)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        return x_p

    def rk3(self, x, u, f, M, h):
        '''
        Discrete nonlinear dynamics (RK3 approx.)
        '''
        x_p = x
        for _ in range(M):
            a1 = h * f(x_p, u)
            a2 = h * f(x_p + a1/2, u)
            a3 = h * f(x_p - a1 + 2*a2, u)
            x_p += (a1 + 4*a2 + a3) / 6
        return x_p

    def rk2(self, x, u, f, M, h):
        x_p = x
        for _ in range(M):
            a1 = f(x_p, u)
            a2 = f(x_p + h*a1, u)
            x_p += h * (a1 + a2) / 2
        return x_p

    def ca_pos_abs(self, x, eps = 1e-3):
        '''
        smooth, positive apporoximation to abs(x)
        meant for tire slip ratios, where result must be nonzero
        '''
        return ca.sqrt(x**2 + eps**2)

    def ca_abs(self, x):
        '''
        absolute value in casadi
        do not use for tire slip ratios
        used for quadratic drag: c * v * abs(v)
        '''
        return ca.if_else(x > 0, x, -x)

    def ca_sign(self, x, eps = 1e-3):
        ''' smooth apporoximation to sign(x)'''
        return x / self.ca_pos_abs(x)

class CasadiIntegrator(CasadiDynamicsModel):
    '''
    Single integrator
    '''
    def __init__(self, t0: float, model_config: DynamicsConfig = DynamicsConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = False

        self.n_q = 1
        self.n_u = 1

        self.use_mx = False

        # symbolic variables
        self.sym_v      = ca.SX.sym('v')
        self.sym_a      = ca.SX.sym('a')

        # time derivatives
        self.sym_dv     = self.sym_a

        # state and state derivative functions
        self.sym_q = self.sym_v
        self.sym_u = self.sym_a
        self.sym_dq = self.sym_dv

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_a, 0, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, 0], self.options('f_ang_a'))

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long])
        u = np.array([state.u.u_a])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long  = q[0]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long  = q[0]
        if u is not None:
            state.u.u_a     = u[0]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.v_long   = array.array('d', q[:, 0])
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
        
        return prediction
    
class CasadiKinematicUnicycle(CasadiDynamicsModel):
    '''
    Global frame of reference kinematic bicycle

    Body frame velocities and global frame positions
    '''
    def __init__(self, t0: float, model_config: DynamicsConfig = DynamicsConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = False

        self.n_q = 4
        self.n_u = 2

        self.m      = self.model_config.mass

        # symbolic variables
        self.sym_x      = ca.SX.sym('x')
        self.sym_y      = ca.SX.sym('y')
        self.sym_v      = ca.SX.sym('v')
        self.sym_psi    = ca.SX.sym('psi')
        self.sym_Fx    = ca.SX.sym('Fx')
        self.sym_wz    = ca.SX.sym('wz')

        # time derivatives
        self.sym_dx     = self.sym_v * ca.cos(self.sym_psi)
        self.sym_dy     = self.sym_v * ca.sin(self.sym_psi)
        self.sym_dv     = self.sym_Fx/self.m
        self.sym_dpsi   = self.sym_wz

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_x, self.sym_y, self.sym_v, self.sym_psi)
        self.sym_u = ca.vertcat(self.sym_Fx, self.sym_wz)
        self.sym_dq = ca.vertcat(self.sym_dx, self.sym_dy, self.sym_dv, self.sym_dpsi)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_Fx/self.m, 0, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, 0], self.options('f_ang_a'))

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.e.psi])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.x.x       = q[0]
        state.x.y       = q[1]
        state.v.v_long  = q[2]
        state.e.psi     = q[3]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.x.x       = q[0]
            state.x.y       = q[1]
            state.v.v_long  = q[2]
            state.e.psi     = q[3]
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.x        = array.array('d', q[:, 0])
            prediction.y        = array.array('d', q[:, 1])
            prediction.v_long   = array.array('d', q[:, 2])
            prediction.psi      = array.array('d', q[:, 3])
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
        
        return prediction

class CasadiKinematicClUnicycle(CasadiDynamicsModel):
    '''
    Frenet frame of reference point mass

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: UnicycleConfig = UnicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()
        self.get_tangent = self.track.get_tangent_angle_casadi_fn()

        self.n_q = 4
        self.n_u = 2

        self.m      = self.model_config.mass
        self.c_dr   = self.model_config.drag_coefficient
        self.c_da   = self.model_config.damping_coefficient
        self.c_r    = self.model_config.rolling_resistance
        self.p_r    = self.model_config.rolling_resistance_exponent

        # symbolic variables
        self.sym_v      = ca.SX.sym('v')
        self.sym_epsi   = ca.SX.sym('epsi')
        self.sym_s      = ca.SX.sym('s')
        self.sym_xtran  = ca.SX.sym('xtran')
        self.sym_ax     = ca.SX.sym('ax')
        self.sym_wz     = ca.SX.sym('wz')

        self.sym_c = self.get_curvature(self.sym_s)

        # time derivatives
        self.sym_dv     = self.sym_ax - self.c_da*self.sym_v/self.m
        self.sym_depsi  = self.sym_wz - self.sym_c * (self.sym_v * ca.cos(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds     = self.sym_v * ca.cos(self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran = self.sym_v * ca.sin(self.sym_epsi)

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_v, self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_ax, self.sym_wz)
        self.sym_dq = ca.vertcat(self.sym_dv, self.sym_depsi, self.sym_ds, self.sym_dxtran)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, 0, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, 0], self.options('f_ang_a'))

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long  = q[0]
        state.p.e_psi   = q[1]
        state.p.s       = q[2]
        state.p.x_tran  = q[3]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long  = q[0]
            state.p.e_psi   = q[1]
            state.p.s       = q[2]
            state.p.x_tran  = q[3]
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.v_long   = array.array('d', q[:,0])
            prediction.e_psi    = array.array('d', q[:,1])
            prediction.s        = array.array('d', q[:,2])
            prediction.x_tran   = array.array('d', q[:,3])
        if u is not None:
            prediction.u_a      = array.array('d', u[:,0])
            prediction.u_steer  = array.array('d', u[:,1])
        
        return prediction

class CasadiKinematicUnicycleCombined(CasadiDynamicsModel):
    '''
    Frenet frame of reference point mass

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: UnicycleConfig = UnicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()
        self.get_tangent = self.track.get_tangent_angle_casadi_fn()

        self.n_q = 6
        self.n_u = 2

        self.m      = self.model_config.mass
        self.c_dr   = self.model_config.drag_coefficient
        self.c_da   = self.model_config.damping_coefficient
        self.c_r    = self.model_config.rolling_resistance
        self.p_r    = self.model_config.rolling_resistance_exponent

        self.use_mx = self.model_config.use_mx
        if isinstance(self.track, CasadiBSplineTrack):
            self.use_mx = True

        if self.use_mx:
            sym = ca.MX.sym
        else:
            sym = ca.SX.sym

        # symbolic variables
        self.sym_x      = sym('x')
        self.sym_y      = sym('y')
        self.sym_v      = sym('v')
        self.sym_epsi   = sym('epsi')
        self.sym_s      = sym('s')
        self.sym_xtran  = sym('xtran')
        self.sym_Fx     = sym('Fx')
        self.sym_wz     = sym('wz')

        self.sym_c = self.get_curvature(self.sym_s)
        self.sym_psi_t = self.get_tangent(self.sym_s)

        # time derivatives
        self.sym_dx     = self.sym_v * ca.cos(self.sym_psi_t + self.sym_epsi)
        self.sym_dy     = self.sym_v * ca.sin(self.sym_psi_t + self.sym_epsi)
        self.sym_dv     = (self.sym_Fx - self.c_da*self.sym_v)/self.m
        self.sym_depsi  = self.sym_wz - self.sym_c * (self.sym_v * ca.cos(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds     = self.sym_v * ca.cos(self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran = self.sym_v * ca.sin(self.sym_epsi)

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_x, self.sym_y, self.sym_v, self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_Fx, self.sym_wz)
        self.sym_dq = ca.vertcat(self.sym_dx, self.sym_dy, self.sym_dv, self.sym_depsi, self.sym_ds, self.sym_dxtran)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_Fx/self.m, 0, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, 0], self.options('f_ang_a'))

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.x.x       = q[0]
        state.x.y       = q[1]
        state.v.v_long  = q[2]
        state.p.e_psi   = q[3]
        state.p.s       = q[4]
        state.p.x_tran  = q[5]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.x.x       = q[0]
            state.x.y       = q[1]
            state.v.v_long  = q[2]
            state.p.e_psi   = q[3]
            state.p.s       = q[4]
            state.p.x_tran  = q[5]
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.x        = array.array('d', q[:,0])
            prediction.y        = array.array('d', q[:,1])
            prediction.v_long   = array.array('d', q[:,2])
            prediction.e_psi    = array.array('d', q[:,3])
            prediction.s        = array.array('d', q[:,4])
            prediction.x_tran   = array.array('d', q[:,5])
        if u is not None:
            prediction.u_a      = array.array('d', u[:,0])
            prediction.u_steer  = array.array('d', u[:,1])
        
        return prediction

class CasadiKinematicBicycle(CasadiDynamicsModel):
    '''
    Global frame of reference kinematic bicycle

    Body frame velocities and global frame positions
    '''
    def __init__(self, t0: float, model_config: KinematicBicycleConfig = KinematicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = False

        self.L_f    = self.model_config.wheel_dist_front
        self.L_r    = self.model_config.wheel_dist_rear

        self.c_dr   = self.model_config.drag_coefficient
        self.c_da   = self.model_config.damping_coefficient
        self.c_s    = self.model_config.slip_coefficient
        self.c_r    = self.model_config.rolling_resistance
        self.p_r    = self.model_config.rolling_resistance_exponent

        self.m      = self.model_config.mass

        self.use_mx = self.model_config.use_mx
        if isinstance(self.track, CasadiBSplineTrack):
            self.use_mx = True

        if self.use_mx:
            sym = ca.MX.sym
        else:
            sym = ca.SX.sym

        self.n_q = 4
        self.n_u = 2

        # symbolic variables
        self.sym_x      = sym('x')
        self.sym_y      = sym('y')
        self.sym_v      = sym('v')
        self.sym_psi    = sym('psi')
        self.sym_u_s    = sym('gamma')
        self.sym_u_a    = sym('a')

        # turning angle
        L = self.L_f + self.L_r
        # self.sym_beta = ca.atan(ca.tan(self.sym_u_s) * self.L_r / L)
        self.sym_beta = ca.atan2(ca.tan(self.sym_u_s) * self.L_r, L)

        # yaw rate
        psidot = self.sym_v / self.L_r * ca.sin(self.sym_beta)

        # External forces
        F_ext = - self.c_da * self.sym_v \
                - self.c_dr * self.sym_v * self.ca_abs(self.sym_v) \
                - self.c_r * self.ca_abs(self.sym_v)**self.p_r * self.ca_sign(self.sym_v) \
                - self.c_s * ca.constpow(psidot, 2)

        # time derivatives
        self.sym_dx     = self.sym_v * ca.cos(self.sym_beta + self.sym_psi)
        self.sym_dy     = self.sym_v * ca.sin(self.sym_beta + self.sym_psi)
        self.sym_dv     = self.sym_u_a + F_ext/self.m
        self.sym_dpsi   = psidot

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_x, self.sym_y, self.sym_v, self.sym_psi)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dx, self.sym_dy, self.sym_dv, self.sym_dpsi)

        self.sym_ax = self.sym_dv * ca.cos(self.sym_beta)
        self.sym_ay = self.sym_dv * ca.sin(self.sym_beta)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_dv/self.L_r*ca.sin(self.sym_beta)], self.options('f_ang_a'))
        
        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.e.psi])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a = u[0]
        input.u_steer = u[1]
        return

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.x.x       = q[0]
        state.x.y       = q[1]
        state.v.v_long  = q[2]
        state.e.psi     = q[3]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.x.x       = q[0]
            state.x.y       = q[1]
            state.v.v_long  = q[2]
            state.e.psi     = q[3]
            if u is not None:
                state.w.w_psi = q[2] / self.L_r * np.sin(np.arctan(np.tan(u[1]) * self.L_f / (self.L_f + self.L_r)))
                state.v.v_tran = state.w.w_psi * self.L_r
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if u is not None:
            psidot = np.multiply(q[:-1, 0] * self.L_r,
                                 np.sin(np.arctan(np.tan(u[:, 1]) * self.L_f / (self.L_f + self.L_r))))
            psidot = np.append(psidot, psidot[-1])
            v_tran = psidot * self.L_r

        if q is not None:
            prediction.x        = array.array('d', q[:, 0])
            prediction.y        = array.array('d', q[:, 1])
            prediction.v_long   = array.array('d', q[:, 2])
            prediction.psi      = array.array('d', q[:, 3])
            if u is not None:
                prediction.psidot = array.array('d', psidot)
                prediction.v_tran = array.array('d', v_tran)
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
        
        return prediction

class CasadiKinematicCLBicycle(CasadiDynamicsModel):
    '''
    Frenet frame of reference kinematic bicycle

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: KinematicBicycleConfig = KinematicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()

        self.L_f    = self.model_config.wheel_dist_front
        self.L_r    = self.model_config.wheel_dist_rear

        self.c_dr   = self.model_config.drag_coefficient
        self.c_da   = self.model_config.damping_coefficient
        self.c_s    = self.model_config.slip_coefficient
        self.c_r    = self.model_config.rolling_resistance
        self.p_r    = self.model_config.rolling_resistance_exponent

        self.m      = self.model_config.mass

        self.n_q = 4
        self.n_u = 2

        # Symbolic state variables
        self.sym_epsi   = ca.SX.sym('epsi')
        self.sym_s      = ca.SX.sym('s')
        self.sym_xtran  = ca.SX.sym('xtran')
        self.sym_v     = ca.SX.sym('v')

        # Symbolic input variables
        self.sym_u_s = ca.SX.sym('gamma')
        self.sym_u_a = ca.SX.sym('a')

        # turning angle
        L = self.L_f + self.L_r
        self.sym_beta = ca.atan2(ca.tan(self.sym_u_s) * self.L_r, L)

        self.sym_psidot = self.sym_v * ca.sin(self.sym_beta) / self.L_r

        F_ext = - self.c_da * self.sym_v \
                - self.c_dr * self.sym_v * self.ca_abs(self.sym_v) \
                - self.c_r * self.ca_abs(self.sym_v)**self.p_r * self.ca_sign(self.sym_v) \
                - self.c_s * ca.constpow(self.sym_psidot, 2)

        self.sym_c = self.get_curvature(self.sym_s)

        # time derivatives
        self.sym_dv         = self.sym_u_a + F_ext/self.m
        self.sym_depsi      = self.sym_psidot - self.sym_c * self.sym_v * ca.cos(self.sym_beta + self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds         = self.sym_v * ca.cos(self.sym_beta + self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran     = self.sym_v * ca.sin(self.sym_beta + self.sym_epsi)

        self.sym_q = ca.vertcat(self.sym_v, self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dv, self.sym_depsi, self.sym_ds, self.sym_dxtran)

        self.sym_ax = self.sym_dv * ca.cos(self.sym_beta)
        self.sym_ay = self.sym_dv * ca.sin(self.sym_beta)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_dv*ca.sin(self.sym_beta)/self.L_r], self.options('f_ang_a'))

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long  = q[0]
        state.p.e_psi   = q[1]
        state.p.s       = q[2]
        state.p.x_tran  = q[3]

        # state.psidot = q[0] / self.L_r * np.sin(np.arctan(np.tan(u[1]) * self.L_f / (self.L_f + self.L_r)))
        # state.v_tran = state.psidot * self.L_r
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long  = q[0]
            state.p.e_psi   = q[1]
            state.p.s       = q[2]
            state.p.x_tran  = q[3]
            if u is not None:
                state.w.w_psi = q[0] / self.L_r * np.sin(np.arctan(np.tan(u[1]) * self.L_f / (self.L_f + self.L_r)))
                state.v.v_tran = state.w.w_psi * self.L_r
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if u is not None:
            psidot = np.multiply(q[:-1, 0] * self.L_r,
                                 np.sin(np.arctan(np.tan(u[:, 1]) * self.L_f / (self.L_f + self.L_r))))
            psidot = np.append(psidot, psidot[-1])
            v_tran = psidot * self.L_r

        if q is not None:
            prediction.v_long   = array.array('d', q[:, 0])
            prediction.e_psi    = array.array('d', q[:, 1])
            prediction.s        = array.array('d', q[:, 2])
            prediction.x_tran   = array.array('d', q[:, 3])
            if u is not None:
                prediction.psidot = array.array('d', psidot)
                prediction.v_tran = array.array('d', v_tran)
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
        return prediction

class CasadiKinematicCLVelBicycle(CasadiDynamicsModel):
    '''
    Frenet frame of reference kinematic bicycle

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: KinematicBicycleConfig = KinematicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()

        self.L_f    = self.model_config.wheel_dist_front
        self.L_r    = self.model_config.wheel_dist_rear

        self.c_dr   = self.model_config.drag_coefficient
        self.c_da   = self.model_config.damping_coefficient
        self.c_s    = self.model_config.slip_coefficient
        self.c_r    = self.model_config.rolling_resistance
        self.p_r    = self.model_config.rolling_resistance_exponent

        self.m      = self.model_config.mass

        self.n_q = 3
        self.n_u = 2

        # Symbolic state variables
        self.sym_epsi   = ca.SX.sym('epsi')
        self.sym_s      = ca.SX.sym('s')
        self.sym_xtran  = ca.SX.sym('xtran')

        # Symbolic input variables
        self.sym_u_s = ca.SX.sym('steer')
        self.sym_u_v = ca.SX.sym('v')

        # turning angle
        L = self.L_f + self.L_r
        self.sym_beta = ca.atan2(ca.tan(self.sym_u_s) * self.L_r, L)

        self.sym_c = self.get_curvature(self.sym_s)

        # time derivatives
        self.sym_depsi      = self.sym_u_v * ca.sin(self.sym_beta) / self.L_r - self.sym_c * self.sym_u_v * ca.cos(self.sym_beta + self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds         = self.sym_u_v * ca.cos(self.sym_beta + self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran     = self.sym_u_v * ca.sin(self.sym_beta + self.sym_epsi)

        self.sym_q = ca.vertcat(self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_v, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_depsi, self.sym_ds, self.sym_dxtran)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [0, 0, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, 0], self.options('f_ang_a'))

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.p.e_psi   = q[0]
        state.p.s       = q[1]
        state.p.x_tran  = q[2]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.p.e_psi   = q[0]
            state.p.s       = q[1]
            state.p.x_tran  = q[2]
            if u is not None:
                beta = np.arctan(np.tan(u[1]) * self.L_r / (self.L_f + self.L_r))
                state.v.v_tran = u[0] * np.sin(beta)
                state.v.v_long = u[0] * np.cos(beta)
                state.w.w_psi = u[0] * np.sin(beta) / self.L_r
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if u is not None:
            beta = np.arctan(np.tan(u[:,1]) * self.L_r / (self.L_f + self.L_r))
            v_tran = u[:,0] * np.sin(beta)
            v_long = u[:,0] * np.cos(beta)
            w_psi = u[:,0] * np.sin(beta) / self.L_r

        if q is not None:
            prediction.e_psi    = array.array('d', q[:, 0])
            prediction.s        = array.array('d', q[:, 1])
            prediction.x_tran   = array.array('d', q[:, 2])
            if u is not None:
                prediction.v_tran = array.array('d', v_tran)
                prediction.v_long = array.array('d', v_long)
                prediction.psidot = array.array('d', w_psi)
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
        return prediction
    
class CasadiKinematicBicycleCombined(CasadiDynamicsModel):
    '''
    Frenet frame of reference kinematic bicycle

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: KinematicBicycleConfig = KinematicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()
        self.get_tangent = self.track.get_tangent_angle_casadi_fn()

        self.L_f    = self.model_config.wheel_dist_front
        self.L_r    = self.model_config.wheel_dist_rear

        self.c_dr   = self.model_config.drag_coefficient
        self.c_da   = self.model_config.damping_coefficient
        self.c_s    = self.model_config.slip_coefficient
        self.c_r    = self.model_config.rolling_resistance
        self.p_r    = self.model_config.rolling_resistance_exponent

        self.m      = self.model_config.mass 

        self.use_mx = self.model_config.use_mx
        if isinstance(self.track, CasadiBSplineTrack):
            self.use_mx = True

        if self.use_mx:
            sym = ca.MX.sym
        else:
            sym = ca.SX.sym

        self.n_q = 6
        self.n_u = 2

        # Symbolic state variables
        self.sym_x      = sym('x')
        self.sym_y      = sym('y')
        self.sym_v      = sym('v')
        self.sym_epsi   = sym('epsi')
        self.sym_s      = sym('s')
        self.sym_xtran  = sym('xtran')

        # Symbolic input variables
        self.sym_u_s = sym('gamma')
        self.sym_u_a = sym('a')

        # turning angle
        L = self.L_f + self.L_r
        # self.sym_beta = ca.atan(ca.tan(self.sym_u_s) * self.L_r / L)
        self.sym_beta   = ca.atan2(ca.tan(self.sym_u_s) * self.L_r, L)

        self.sym_psidot = self.sym_v / self.L_r * ca.sin(self.sym_beta)

        F_ext = - self.c_da * self.sym_v \
                - self.c_dr * self.sym_v * self.ca_abs(self.sym_v) \
                - self.c_r * self.ca_abs(self.sym_v)**self.p_r * self.ca_sign(self.sym_v) \
                - self.c_s * ca.constpow(self.sym_psidot, 2)

        self.sym_c      = self.get_curvature(self.sym_s)
        self.sym_psi_t  = self.get_tangent(self.sym_s)

        # time derivatives
        self.sym_dx         = self.sym_v * ca.cos(self.sym_beta + self.sym_psi_t + self.sym_epsi)
        self.sym_dy         = self.sym_v * ca.sin(self.sym_beta + self.sym_psi_t + self.sym_epsi)
        self.sym_dv         = self.sym_u_a + F_ext/self.m
        self.sym_depsi      = self.sym_psidot - self.sym_c * self.sym_v * ca.cos(self.sym_beta + self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds         = self.sym_v * ca.cos(self.sym_beta + self.sym_epsi) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran     = self.sym_v * ca.sin(self.sym_beta + self.sym_epsi)

        self.sym_q  = ca.vertcat(self.sym_x, self.sym_y, self.sym_v, self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u  = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dx, self.sym_dy, self.sym_dv, self.sym_depsi, self.sym_ds, self.sym_dxtran)

        self.sym_ax = self.sym_dv * ca.cos(self.sym_beta)
        self.sym_ay = self.sym_dv * ca.sin(self.sym_beta)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_dv/self.L_r*ca.sin(self.sym_beta)], self.options('f_ang_a'))
        
        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.x.x       = q[0]
        state.x.y       = q[1]
        state.v.v_long  = q[2]
        state.p.e_psi   = q[3]
        state.p.s       = q[4]
        state.p.x_tran  = q[5]

        # state.psidot = q[0] / self.L_r * np.sin(np.arctan(np.tan(u[1]) * self.L_f / (self.L_f + self.L_r)))
        # state.v_tran = state.psidot * self.L_r
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.x.x       = q[0]
            state.x.y       = q[1]
            state.v.v_long  = q[2]
            state.p.e_psi   = q[3]
            state.p.s       = q[4]
            state.p.x_tran  = q[5]
            if u is not None:
                state.w.w_psi = q[2] / self.L_r * np.sin(np.arctan(np.tan(u[1]) * self.L_f / (self.L_f + self.L_r)))
                state.v.v_tran = state.w.w_psi * self.L_r
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if u is not None:
            psidot = np.multiply(q[:-1, 2] * self.L_r,
                                 np.sin(np.arctan(np.tan(u[:, 1]) * self.L_f / (self.L_f + self.L_r))))
            psidot = np.append(psidot, psidot[-1])
            v_tran = psidot * self.L_r

        if q is not None:
            prediction.x        = array.array('d', q[:, 0])
            prediction.y        = array.array('d', q[:, 1])
            prediction.v_long   = array.array('d', q[:, 2])
            prediction.e_psi    = array.array('d', q[:, 3])
            prediction.s        = array.array('d', q[:, 4])
            prediction.x_tran   = array.array('d', q[:, 5])
            if u is not None:
                prediction.psidot = array.array('d', psidot)
                prediction.v_tran = array.array('d', v_tran)
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
        
        return prediction
    
class CasadiKinematicBicycleProgressAugmented(CasadiDynamicsModel):
    '''
    Global frame of reference kinematic bicycle

    Body frame velocities and global frame positions
    '''
    def __init__(self, t0: float, 
                 model_config: KinematicBicycleConfig = KinematicBicycleConfig(), 
                 track=None, 
                 track_tightening=0):
        super().__init__(t0, model_config, track=track)
        self.curvature_model = False

        self.L_f    = self.model_config.wheel_dist_front
        self.L_r    = self.model_config.wheel_dist_rear

        self.c_dr   = self.model_config.drag_coefficient
        self.c_da   = self.model_config.damping_coefficient
        self.c_s    = self.model_config.slip_coefficient
        self.c_r    = self.model_config.rolling_resistance
        self.p_r    = self.model_config.rolling_resistance_exponent

        self.m      = self.model_config.mass

        self.code_gen = self.model_config.code_gen
        self.opt_flag = self.model_config.opt_flag

        self.use_mx = self.model_config.use_mx
        if self.use_mx:
            sym = ca.MX.sym
        else:
            sym = ca.SX.sym

        self.n_q = 5
        self.n_u = 3

        # symbolic variables
        self.sym_x      = sym('x')
        self.sym_y      = sym('y')
        self.sym_v      = sym('v')
        self.sym_psi    = sym('psi')
        self.sym_s      = sym('s')
        self.sym_u_s    = sym('steer')
        self.sym_u_a    = sym('accel')
        self.sym_u_ds   = sym('ds')

        # turning angle
        L = self.L_f + self.L_r
        # self.sym_beta = ca.atan(ca.tan(self.sym_u_s) * self.L_r / L)
        self.sym_beta = ca.atan2(ca.tan(self.sym_u_s) * self.L_r, L)

        # yaw rate
        psidot = self.sym_v / self.L_r * ca.sin(self.sym_beta)

        # External forces
        F_ext = - self.c_da * self.sym_v \
                - self.c_dr * self.sym_v * self.ca_abs(self.sym_v) \
                - self.c_r * self.ca_abs(self.sym_v)**self.p_r * self.ca_sign(self.sym_v) \
                - self.c_s * ca.constpow(psidot, 2)

        # time derivatives
        self.sym_dx     = self.sym_v * ca.cos(self.sym_beta + self.sym_psi)
        self.sym_dy     = self.sym_v * ca.sin(self.sym_beta + self.sym_psi)
        self.sym_dv     = self.sym_u_a + F_ext/self.m
        self.sym_dpsi   = psidot
        self.sym_ds     = self.sym_u_ds

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_x, self.sym_y, self.sym_v, self.sym_psi, self.sym_s)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s, self.sym_u_ds)
        self.sym_dq = ca.vertcat(self.sym_dx, self.sym_dy, self.sym_dv, self.sym_dpsi, self.sym_ds)

        self.sym_ax = self.sym_dv * ca.cos(self.sym_beta)
        self.sym_ay = self.sym_dv * ca.sin(self.sym_beta)

        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_dv/self.L_r*ca.sin(self.sym_beta)], self.options('f_ang_a'))
        
        self.precompute_model()

        if isinstance(self.track, RadiusArclengthTrack):
            # Compute spline approximation of track
            S = np.linspace(0, self.track.track_length, 100)
            X, Y, Xi, Yi, Xo, Yo = [], [], [], [], [], []
            for s in S:
                # Centerline
                x, y, _ = self.track.local_to_global((s, 0, 0))
                X.append(x)
                Y.append(y)
                # Inside boundary
                xi, yi, _ = self.track.local_to_global((s, self.track.half_width-track_tightening, 0))
                Xi.append(xi)
                Yi.append(yi)
                # Outside boundary
                xo, yo, _ = self.track.local_to_global((s, -(self.track.half_width-track_tightening), 0))
                Xo.append(xo)
                Yo.append(yo)
            self.x_s = ca.interpolant('x_s', 'bspline', [S], X)
            self.y_s = ca.interpolant('y_s', 'bspline', [S], Y)
            self.xi_s = ca.interpolant('xi_s', 'bspline', [S], Xi)
            self.yi_s = ca.interpolant('yi_s', 'bspline', [S], Yi)
            self.xo_s = ca.interpolant('xo_s', 'bspline', [S], Xo)
            self.yo_s = ca.interpolant('yo_s', 'bspline', [S], Yo)

            # Compute derivatives of track
            s_sym = ca.MX.sym('s', 1)
            self.dxds = ca.Function('dxds', [s_sym], [ca.jacobian(self.x_s(s_sym), s_sym)])
            self.dyds = ca.Function('dyds', [s_sym], [ca.jacobian(self.y_s(s_sym), s_sym)])
        elif isinstance(self.track, CasadiBSplineTrack):
            self.x_s = self.track.x
            self.y_s = self.track.y
            self.xi_s = self.track.xi
            self.yi_s = self.track.yi
            self.xo_s = self.track.xo
            self.yo_s = self.track.yo

            self.dxds = self.track.dx
            self.dyds = self.track.dy
        else:
            raise(ValueError(f'Track type {type(self.track)} not supported'))

        self.pos_idx = [0, 1]
        return

    def get_contouring_lag_costs(self, contouring_cost, lag_cost):
        sym_q = ca.MX.sym('q', self.n_q)
        L = self.track.track_length
        # Contouring and lag errors and their gradients
        # s_mod = ca.fmod(sym_q[-1], L)
        s_mod = ca.fmod(ca.fmod(sym_q[-1], L) + L, L)
        # Reference interpolation variable must be in range [-1, 1] (outside, inside)
        z_sym = ca.MX.sym('z', 1)
        t = ca.atan2(self.dyds(s_mod), self.dxds(s_mod))
        x_int = self.xo_s(s_mod) + (z_sym+1)/2*(self.xi_s(s_mod)-self.xo_s(s_mod))
        y_int = self.yo_s(s_mod) + (z_sym+1)/2*(self.yi_s(s_mod)-self.yo_s(s_mod))
        ec =  ca.sin(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.cos(t)*(sym_q[self.pos_idx[1]]-y_int)
        el = -ca.cos(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.sin(t)*(sym_q[self.pos_idx[1]]-y_int)
        f_e = ca.Function('ec', [sym_q, z_sym], [ca.vertcat(contouring_cost*ec, lag_cost*el)])
        return f_e

    def get_contouring_lag_costs_quad_approx(self, contouring_cost, lag_cost):
        sym_q = ca.MX.sym('q', self.n_q)
        L = self.track.track_length
        # Contouring and lag errors and their gradients
        # s_mod = ca.fmod(sym_q[-1], L)
        s_mod = ca.fmod(ca.fmod(sym_q[-1], L) + L, L)
        # Reference interpolation variable must be in range [-1, 1] (outside, inside)
        z_sym = ca.MX.sym('z', 1)
        t = ca.atan2(self.dyds(s_mod), self.dxds(s_mod))
        x_int = self.xo_s(s_mod) + (z_sym+1)/2*(self.xi_s(s_mod)-self.xo_s(s_mod))
        y_int = self.yo_s(s_mod) + (z_sym+1)/2*(self.yi_s(s_mod)-self.yo_s(s_mod))
        ec =  ca.sin(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.cos(t)*(sym_q[self.pos_idx[1]]-y_int)
        el = -ca.cos(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.sin(t)*(sym_q[self.pos_idx[1]]-y_int)
        
        e = ca.vertcat(ec, el)
        Dx_e = ca.jacobian(e, sym_q)

        P_cl = ca.diag(ca.vertcat(contouring_cost, lag_cost))

        # The approximation is (1/2) q.T @ Q_e(q_bar) @ q + q_e(q_bar).T @ q
        Q_e = Dx_e.T @ P_cl @ Dx_e
        q_e = Dx_e.T @ P_cl @ e - Q_e @ sym_q
        
        options = dict(jit=self.code_gen, jit_name='contouring_lag_approx', compiler='shell', jit_options=dict(compiler='gcc', flags=[f'-{self.opt_flag}'], verbose=False))
        f_cl = ca.Function('contouring_lag_approx', [sym_q, z_sym], [Q_e, q_e], options)

        return f_cl

    def get_track_boundary_constraint_lin_approx(self):
        sym_q = ca.MX.sym('q', self.n_q)
        L = self.track.track_length
        # s_mod = ca.fmod(sym_q[-1], L)
        s_mod = ca.fmod(ca.fmod(sym_q[-1], L) + L, L)

        # Linear approximation of track boundary constraints
        xi, yi = self.xi_s(s_mod), self.yi_s(s_mod)
        xo, yo = self.xo_s(s_mod), self.yo_s(s_mod)
        n, d = -(xo - xi), yo - yi

        # The approximation is G @ x + g <= 0
        G = ca.MX.sym('G', ca.Sparsity(2, self.n_q))
        G[0,self.pos_idx[0]], G[0,self.pos_idx[1]] = n, -d
        G[1,self.pos_idx[0]], G[1,self.pos_idx[1]] = -n, d
        g = ca.vertcat(-ca.fmax(n*xi-d*yi, n*xo-d*yo), ca.fmin(n*xi-d*yi, n*xo-d*yo))

        options = dict(jit=self.code_gen, jit_name='track_boundary_approx', compiler='shell', jit_options=dict(compiler='gcc', flags=[f'-{self.opt_flag}'], verbose=False))
        f_tb = ca.Function('G', [sym_q], [G, g], options)

        return f_tb

    def get_arcspeed_cost(self, magnitude_weight, performance_weight):
        sym_u = ca.SX.sym('u', self.n_u)
        arcspeed_cost = (1/2)*magnitude_weight*sym_u[-1]**2 - performance_weight*sym_u[-1]
        f_u = ca.Function('u', [sym_u], [arcspeed_cost])
        return f_u

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.e.psi, state.p.s])
        u = np.array([state.u.u_a, state.u.u_steer, state.u.u_ds])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.e.psi, state.p.s])
        return q

    def input2u(self, control: VehicleActuation) -> np.ndarray:
        u = np.array([control.u_a, control.u_steer, control.u_ds])
        return u

    def u2input(self, control: VehicleActuation, u: np.ndarray):
        control.u_a       = u[0]
        control.u_steer   = u[1]
        control.u_ds      = u[2]
        return

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.x.x       = q[0]
        state.x.y       = q[1]
        state.v.v_long  = q[2]
        state.e.psi     = q[3]
        state.p.s       = q[4]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.x.x       = q[0]
            state.x.y       = q[1]
            state.v.v_long  = q[2]
            state.e.psi     = q[3]
            state.p.s       = q[4]
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
            state.u.u_ds    = u[2]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if u is not None:
            psidot = np.multiply(q[:-1, 0] * self.L_r,
                                 np.sin(np.arctan(np.tan(u[:, 1]) * self.L_f / (self.L_f + self.L_r))))
            psidot = np.append(psidot, psidot[-1])
            v_tran = psidot * self.L_r

        if q is not None:
            prediction.x        = array.array('d', q[:, 0])
            prediction.y        = array.array('d', q[:, 1])
            prediction.v_long   = array.array('d', q[:, 2])
            prediction.psi      = array.array('d', q[:, 3])
            prediction.s        = array.array('d', q[:, 4])
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
            prediction.u_ds     = array.array('d', u[:, 2])
        
        return prediction

class CasadiDynamicBicycle(CasadiDynamicsModel):
    '''
    Global frame of reference dynamic bicycle model - Pacejka model tire forces

    Body frame velocities and global frame positions
    '''

    def __init__(self, t0: float,
                    model_config: DynamicBicycleConfig = DynamicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = False

        self.n_q = 6
        self.n_u = 2

        self.L_f            = self.model_config.wheel_dist_front
        self.L_r            = self.model_config.wheel_dist_rear

        self.m              = self.model_config.mass
        self.I_z            = self.model_config.yaw_inertia
        self.g              = self.model_config.gravity

        self.c_dr           = self.model_config.drag_coefficient
        self.c_da           = self.model_config.damping_coefficient
        self.c_r            = self.model_config.rolling_resistance
        self.p_r            = self.model_config.rolling_resistance_exponent

        self.mu             = self.model_config.wheel_friction
        self.tire_model     = self.model_config.tire_model
        self.drive_wheels   = self.model_config.drive_wheels

        self.pacejka_Bf     = self.model_config.pacejka_b_front
        self.pacejka_Br     = self.model_config.pacejka_b_rear
        self.pacejka_Cf     = self.model_config.pacejka_c_front
        self.pacejka_Cr     = self.model_config.pacejka_c_rear
        self.pacejka_Df     = self.model_config.pacejka_d_front
        self.pacejka_Dr     = self.model_config.pacejka_d_rear

        self.linear_Bf      = self.model_config.linear_bf
        self.linear_Br      = self.model_config.linear_br

        self.simple_slip    = self.model_config.simple_slip

        self.use_mx         = self.model_config.use_mx
        if isinstance(self.track, CasadiBSplineTrack):
            self.use_mx = True

        if self.use_mx:
            sym = ca.MX.sym
        else:
            sym = ca.SX.sym

        # symbolic variables
        self.sym_vx         = sym('vx')  # body fram vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_vy         = sym('vy')
        self.sym_psidot     = sym('psidot')
        self.sym_x          = sym('x')
        self.sym_y          = sym('y')
        self.sym_psi        = sym('psi')
        self.sym_u_s        = sym('gamma')
        self.sym_u_a        = sym('a')

        # Slip angles and Pacejka tire forces
        if self.simple_slip:
            self.sym_alpha_f = -ca.atan2(self.sym_vy + self.L_f * self.sym_psidot, self.sym_vx) + self.sym_u_s
        else:
            self.sym_alpha_f = -ca.atan2((self.sym_vy + self.L_f * self.sym_psidot) * ca.cos(self.sym_u_s) - self.sym_vx * ca.sin(self.sym_u_s),
                                         self.sym_vx * ca.cos(self.sym_u_s) + (self.sym_vy + self.L_f * self.sym_psidot) * ca.sin(self.sym_u_s))
        self.sym_alpha_r = -ca.atan2(self.sym_vy - self.L_r * self.sym_psidot, self.sym_vx)

        if self.tire_model == 'pacejka':
            self.sym_fyf = self.pacejka_Df * ca.sin(self.pacejka_Cf * ca.atan(self.pacejka_Bf * self.sym_alpha_f))
            self.sym_fyr = self.pacejka_Dr * ca.sin(self.pacejka_Cr * ca.atan(self.pacejka_Br * self.sym_alpha_r))
        elif self.tire_model == 'linear':
            self.sym_fyf = self.linear_Bf * self.m*self.g*self.L_r/(self.L_f+self.L_r) * self.sym_alpha_f
            self.sym_fyr = self.linear_Br * self.m*self.g*self.L_f/(self.L_f+self.L_r) * self.sym_alpha_r
        else:
            raise(ValueError("Tire model must be 'linear' or 'pacejka'"))

        # External forces
        F_ext = - self.c_da * self.sym_vx \
                - self.c_dr * self.sym_vx * self.ca_abs(self.sym_vx) \
                - self.c_r * self.ca_abs(self.sym_vx)**self.p_r * self.ca_sign(self.sym_vx)

        if self.drive_wheels == 'all':
            _ar = self.sym_u_a/2
            _af = self.sym_u_a/2
        elif self.drive_wheels == 'rear':
            _ar = self.sym_u_a
            _af = 0
        # instantaneous accelerations
        # self.sym_ax = self.sym_u_a + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        # self.sym_ay = (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_ax = _ar + _af*ca.cos(self.sym_u_s) + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        self.sym_ay = _af*ca.sin(self.sym_u_s) + (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_alphaz = (self.L_f * self.sym_fyf * ca.cos(self.sym_u_s) - self.L_r * self.sym_fyr) / self.I_z

        # time derivatives
        self.sym_dvx        = self.sym_ax + self.sym_psidot * self.sym_vy
        self.sym_dvy        = self.sym_ay - self.sym_psidot * self.sym_vx
        self.sym_dpsidot    = self.sym_alphaz
        self.sym_dx         = self.sym_vx * ca.cos(self.sym_psi) - self.sym_vy * ca.sin(self.sym_psi)
        self.sym_dy         = self.sym_vy * ca.cos(self.sym_psi) + self.sym_vx * ca.sin(self.sym_psi)
        self.sym_dpsi       = self.sym_psidot

        # state and state derivative functions
        self.sym_q  = ca.vertcat(self.sym_vx, self.sym_vy, self.sym_psidot, self.sym_x, self.sym_y, self.sym_psi)
        self.sym_u  = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_dx, self.sym_dy, self.sym_dpsi)
        
        # Auxilliary functions
        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_alpha = ca.Function('f_alpha', [self.sym_q, self.sym_u], [self.sym_alpha_f, self.sym_alpha_r])
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_alphaz], self.options('f_ang_a'))

        self.precompute_model()
        return

    def get_slip_angles(self, state: VehicleState) -> np.ndarray:
        q, u = self.state2qu(state)
        return np.array(self.f_alpha(q, u)).squeeze()

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long  = q[0]
        state.v.v_tran  = q[1]
        state.w.w_psi   = q[2]
        state.x.x       = q[3]
        state.x.y       = q[4]
        state.e.psi     = q[5]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long  = q[0]
            state.v.v_tran  = q[1]
            state.w.w_psi   = q[2]
            state.x.x       = q[3]
            state.x.y       = q[4]
            state.e.psi     = q[5]
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.v_long   = array.array('d', q[:, 0])
            prediction.v_tran   = array.array('d', q[:, 1])
            prediction.psidot   = array.array('d', q[:, 2])
            prediction.x        = array.array('d', q[:, 3])
            prediction.y        = array.array('d', q[:, 4])
            prediction.psi      = array.array('d', q[:, 5])
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])

        return prediction

    def qu2interpolator(self, t: float, q: np.ndarray = None, u: np.ndarray = None, extrapolate: bool = False):
        prediction = VehiclePrediction(t=t)
        if q is not None:
            _t = t + self.dt*np.arange(q.shape[0])

            if extrapolate:
                prediction.v_long   = interp1d(_t, q[:,0], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.v_tran   = interp1d(_t, q[:,1], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.psidot   = interp1d(_t, q[:,2], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.x        = interp1d(_t, q[:,3], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.y        = interp1d(_t, q[:,4], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.psi      = interp1d(_t, q[:,5], kind='linear', assume_sorted=True, fill_value='extrapolate')
            else:
                _v_long     = interp1d(_t, q[:,0], kind='linear', assume_sorted=True)
                _v_tran     = interp1d(_t, q[:,1], kind='linear', assume_sorted=True)
                _psidot     = interp1d(_t, q[:,2], kind='linear', assume_sorted=True)
                _x          = interp1d(_t, q[:,3], kind='linear', assume_sorted=True)
                _y          = interp1d(_t, q[:,4], kind='linear', assume_sorted=True)
                _psi        = interp1d(_t, q[:,5], kind='linear', assume_sorted=True)

                prediction.v_long   = lambda t: _v_long(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.v_tran   = lambda t: _v_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.psidot   = lambda t: _psidot(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.x        = lambda t: _x(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.y        = lambda t: _y(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.psi      = lambda t: _psi(np.maximum(np.minimum(t, _t[-1]), _t[0]))

        if u is not None:
            _t = t + self.dt*np.arange(u.shape[0])
            _u_a        = interp1d(_t, u[:,0], kind='linear', assume_sorted=True)
            _u_steer    = interp1d(_t, u[:,1], kind='linear', assume_sorted=True)

            prediction.u_a      = lambda t: _u_a(np.maximum(np.minimum(t, _t[-1]), _t[0]))
            prediction.u_steer  = lambda t: _u_steer(np.maximum(np.minimum(t, _t[-1]), _t[0]))
        
        return prediction
    
    def prediction2interpolator(self, pred: VehiclePrediction, extrapolate: bool = False):
            interpolator = VehiclePrediction(t=pred.t)

            if None in [pred.v_long, pred.v_tran, pred.psidot, pred.x, pred.y, pred.psi, pred.u_a, pred.u_steer]:
                return None
            
            if extrapolate:
                interpolator.v_long   = interp1d(pred.t + self.dt*np.arange(len(pred.v_long)), pred.v_long, kind='linear', assume_sorted=True, fill_value='extrapolate')
                interpolator.v_tran   = interp1d(pred.t + self.dt*np.arange(len(pred.v_tran)), pred.v_tran, kind='linear', assume_sorted=True, fill_value='extrapolate')
                interpolator.psidot   = interp1d(pred.t + self.dt*np.arange(len(pred.psidot)), pred.psidot, kind='linear', assume_sorted=True, fill_value='extrapolate')
                interpolator.x        = interp1d(pred.t + self.dt*np.arange(len(pred.x)), pred.x, kind='linear', assume_sorted=True, fill_value='extrapolate')
                interpolator.y        = interp1d(pred.t + self.dt*np.arange(len(pred.y)), pred.y, kind='linear', assume_sorted=True, fill_value='extrapolate')
                interpolator.psi      = interp1d(pred.t + self.dt*np.arange(len(pred.psi)), pred.psi, kind='linear', assume_sorted=True, fill_value='extrapolate')
            else:
                _t = pred.t + self.dt*np.arange(len(pred.v_long))
                _v_long = interp1d(_t, pred.v_long, kind='linear', assume_sorted=True)
                interpolator.v_long = lambda t: _v_long(np.maximum(np.minimum(t, _t[-1]), _t[0]))

                _t = pred.t + self.dt*np.arange(len(pred.v_tran))
                _v_tran = interp1d(_t, pred.v_tran, kind='linear', assume_sorted=True)
                interpolator.v_tran = lambda t: _v_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))

                _t = pred.t + self.dt*np.arange(len(pred.psidot))
                _psidot = interp1d(_t, pred.psidot, kind='linear', assume_sorted=True)
                interpolator.psidot = lambda t: _psidot(np.maximum(np.minimum(t, _t[-1]), _t[0]))

                _t = pred.t + self.dt*np.arange(len(pred.x))
                _x = interp1d(_t, pred.x, kind='linear', assume_sorted=True)
                interpolator.x = lambda t: _x(np.maximum(np.minimum(t, _t[-1]), _t[0]))

                _t = pred.t + self.dt*np.arange(len(pred.y))
                _y = interp1d(_t, pred.y, kind='linear', assume_sorted=True)
                interpolator.y = lambda t: _y(np.maximum(np.minimum(t, _t[-1]), _t[0]))

                _t = pred.t + self.dt*np.arange(len(pred.psi))
                _psi = interp1d(_t, pred.psi, kind='linear', assume_sorted=True)
                interpolator.psi = lambda t: _psi(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            _t = pred.t + self.dt*np.arange(len(pred.u_a))
            _u_a        = interp1d(_t, pred.u_a, kind='linear', assume_sorted=True)
            interpolator.u_a      = lambda t: _u_a(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            _t = pred.t + self.dt*np.arange(len(pred.u_steer))
            _u_steer    = interp1d(_t, pred.u_steer, kind='linear', assume_sorted=True)
            interpolator.u_steer  = lambda t: _u_steer(np.maximum(np.minimum(t, _t[-1]), _t[0]))
            
            return interpolator
            
class CasadiDynamicCLBicycle(CasadiDynamicsModel):
    '''
    Frenet frame of reference dynamic bicycle model - Pacejka model tire forces

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: DynamicBicycleConfig = DynamicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()

        self.n_q = 6
        self.n_u = 2

        self.L_f            = self.model_config.wheel_dist_front
        self.L_r            = self.model_config.wheel_dist_rear

        self.m              = self.model_config.mass
        self.I_z            = self.model_config.yaw_inertia
        self.g              = self.model_config.gravity

        self.c_dr           = self.model_config.drag_coefficient
        self.c_da           = self.model_config.damping_coefficient
        self.c_r            = self.model_config.rolling_resistance
        self.p_r            = self.model_config.rolling_resistance_exponent
        
        self.mu             = self.model_config.wheel_friction
        self.tire_model     = self.model_config.tire_model
        self.drive_wheels   = self.model_config.drive_wheels

        self.pacejka_Bf     = self.model_config.pacejka_b_front
        self.pacejka_Br     = self.model_config.pacejka_b_rear
        self.pacejka_Cf     = self.model_config.pacejka_c_front
        self.pacejka_Cr     = self.model_config.pacejka_c_rear
        self.pacejka_Df     = self.model_config.pacejka_d_front
        self.pacejka_Dr     = self.model_config.pacejka_d_rear

        self.linear_Bf      = self.model_config.linear_bf
        self.linear_Br      = self.model_config.linear_br

        self.simple_slip    = self.model_config.simple_slip

        self.use_mx         = self.model_config.use_mx
        if isinstance(self.track, CasadiBSplineTrack):
            self.use_mx = True

        if self.use_mx:
            sym = ca.MX.sym
        else:
            sym = ca.SX.sym

        # symbolic variables
        self.sym_vx     = sym('vx')
        self.sym_vy     = sym('vy')
        self.sym_psidot = sym('psidot')
        self.sym_epsi   = sym('epsi')
        self.sym_s      = sym('s')
        self.sym_xtran  = sym('xtran')
        self.sym_u_s    = sym('gamma')
        self.sym_u_a    = sym('a')

        self.sym_c = self.get_curvature(self.sym_s)

        # Slip angles and Pacejka tire forces
        if self.simple_slip:
            self.sym_alpha_f = -ca.atan2(self.sym_vy + self.L_f * self.sym_psidot, self.sym_vx) + self.sym_u_s
        else:
            self.sym_alpha_f = -ca.atan2(
                (self.sym_vy + self.L_f * self.sym_psidot) * ca.cos(self.sym_u_s) - self.sym_vx * ca.sin(self.sym_u_s),
                self.sym_vx * ca.cos(self.sym_u_s) + (self.sym_vy + self.L_f * self.sym_psidot) * ca.sin(self.sym_u_s))
        self.sym_alpha_r = -ca.atan2(self.sym_vy - self.L_r * self.sym_psidot, self.sym_vx)

        if self.tire_model == 'pacejka':
            self.sym_fyf = self.pacejka_Df * ca.sin(self.pacejka_Cf * ca.atan(self.pacejka_Bf * self.sym_alpha_f))
            self.sym_fyr = self.pacejka_Dr * ca.sin(self.pacejka_Cr * ca.atan(self.pacejka_Br * self.sym_alpha_r))
        elif self.tire_model == 'linear':
            self.sym_fyf = self.linear_Bf * self.m*self.g*self.L_r/(self.L_f+self.L_r) * self.sym_alpha_f
            self.sym_fyr = self.linear_Br * self.m*self.g*self.L_f/(self.L_f+self.L_r) * self.sym_alpha_r
        else:
            raise(ValueError("Tire model must be 'linear' or 'pacejka'"))

        # External forces
        F_ext = - self.c_da * self.sym_vx \
                - self.c_dr * self.sym_vx * self.ca_abs(self.sym_vx) \
                - self.c_r * self.ca_abs(self.sym_vx)**self.p_r * self.ca_sign(self.sym_vx)

        if self.drive_wheels == 'all':
            _ar = self.sym_u_a/2
            _af = self.sym_u_a/2
        elif self.drive_wheels == 'rear':
            _ar = self.sym_u_a
            _af = 0
        # instantaneous accelerations
        # self.sym_ax = self.sym_u_a + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        # self.sym_ay = (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_ax = _ar + _af*ca.cos(self.sym_u_s) + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        self.sym_ay = _af*ca.sin(self.sym_u_s) + (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m        
        self.sym_alphaz = (self.L_f * self.sym_fyf * ca.cos(self.sym_u_s) - self.L_r * self.sym_fyr) / self.I_z

        # time derivatives
        self.sym_dvx        = self.sym_ax + self.sym_psidot * self.sym_vy
        self.sym_dvy        = self.sym_ay - self.sym_psidot * self.sym_vx
        self.sym_dpsidot    = self.sym_alphaz
        self.sym_depsi      = self.sym_psidot - self.sym_c * (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds         = (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran     = self.sym_vx * ca.sin(self.sym_epsi) + self.sym_vy * ca.cos(self.sym_epsi)

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_vx, self.sym_vy, self.sym_psidot, self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_depsi, self.sym_ds, self.sym_dxtran)

        # Auxilliary functions
        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_alpha = ca.Function('f_alpha', [self.sym_q, self.sym_u], [self.sym_alpha_f, self.sym_alpha_r])
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_alphaz], self.options('f_ang_a'))

        ayf = self.sym_fyf/(self.m*self.L_r/(self.L_f+self.L_r))
        ayr = self.sym_fyr/(self.m*self.L_f/(self.L_f+self.L_r))
        self.f_tire_ay = ca.Function('f_tire_ay', [self.sym_q, self.sym_u], [ayf, ayr])

        self.precompute_model()
        return

    def get_slip_angles(self, state: VehicleState) -> np.ndarray:
        q, u = self.state2qu(state)
        return np.array(self.f_alpha(q, u)).squeeze()

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long = q[0]
        state.v.v_tran = q[1]
        state.w.w_psi = q[2]
        state.p.e_psi = q[3]
        state.p.s = q[4]
        state.p.x_tran = q[5]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a = u[0]
        input.u_steer = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long = q[0]
            state.v.v_tran = q[1]
            state.w.w_psi = q[2]
            state.p.e_psi = q[3]
            state.p.s = q[4]
            state.p.x_tran = q[5]
        if u is not None:
            state.u.u_a = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.v_long = array.array('d', q[:, 0])
            prediction.v_tran = array.array('d', q[:, 1])
            prediction.psidot = array.array('d', q[:, 2])
            prediction.e_psi = array.array('d', q[:, 3])
            prediction.s = array.array('d', q[:, 4])
            prediction.x_tran = array.array('d', q[:, 5])
        if u is not None:
            prediction.u_a = array.array('d', u[:, 0])
            prediction.u_steer = array.array('d', u[:, 1])
        
        return prediction
    
    def qu2interpolator(self, t: float, q: np.ndarray = None, u: np.ndarray = None, extrapolate: bool = False):
        prediction = VehiclePrediction(t=t)
        if q is not None:
            _t = t + self.dt*np.arange(q.shape[0])

            if extrapolate:
                prediction.v_long     = interp1d(_t, q[:,0], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.v_tran     = interp1d(_t, q[:,1], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.psidot     = interp1d(_t, q[:,2], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.e_psi      = interp1d(_t, q[:,3], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.s          = interp1d(_t, q[:,4], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.x_tran     = interp1d(_t, q[:,5], kind='linear', assume_sorted=True, fill_value='extrapolate')
            else:
                _v_long     = interp1d(_t, q[:,0], kind='linear', assume_sorted=True)
                _v_tran     = interp1d(_t, q[:,1], kind='linear', assume_sorted=True)
                _psidot     = interp1d(_t, q[:,2], kind='linear', assume_sorted=True)
                _e_psi      = interp1d(_t, q[:,3], kind='linear', assume_sorted=True)
                _s          = interp1d(_t, q[:,4], kind='linear', assume_sorted=True)
                _x_tran     = interp1d(_t, q[:,5], kind='linear', assume_sorted=True)

                prediction.v_long   = lambda t: _v_long(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.v_tran   = lambda t: _v_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.psidot   = lambda t: _psidot(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.e_psi    = lambda t: _e_psi(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.s        = lambda t: _s(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.x_tran   = lambda t: _x_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            
        if u is not None:
            _t = t + self.dt*np.arange(u.shape[0])
            _u_a        = interp1d(_t, u[:,0], kind='linear', assume_sorted=True)
            _u_steer    = interp1d(_t, u[:,1], kind='linear', assume_sorted=True)

            prediction.u_a      = lambda t: _u_a(np.maximum(np.minimum(t, _t[-1]), _t[0]))
            prediction.u_steer  = lambda t: _u_steer(np.maximum(np.minimum(t, _t[-1]), _t[0]))
        
        return prediction

    def prediction2interpolator(self, pred: VehiclePrediction, extrapolate: bool = False):
        interpolator = VehiclePrediction(t=pred.t)

        if None in [pred.v_long, pred.v_tran, pred.psidot, pred.e_psi, pred.s, pred.x_tran, pred.u_a, pred.u_steer]:
            return None
        
        if extrapolate:
            interpolator.v_long   = interp1d(pred.t + self.dt*np.arange(len(pred.v_long)), pred.v_long, kind='linear', assume_sorted=True, fill_value='extrapolate')
            interpolator.v_tran   = interp1d(pred.t + self.dt*np.arange(len(pred.v_tran)), pred.v_tran, kind='linear', assume_sorted=True, fill_value='extrapolate')
            interpolator.psidot   = interp1d(pred.t + self.dt*np.arange(len(pred.psidot)), pred.psidot, kind='linear', assume_sorted=True, fill_value='extrapolate')
            interpolator.e_psi    = interp1d(pred.t + self.dt*np.arange(len(pred.e_psi)), pred.e_psi, kind='linear', assume_sorted=True, fill_value='extrapolate')
            interpolator.s        = interp1d(pred.t + self.dt*np.arange(len(pred.s)), pred.s, kind='linear', assume_sorted=True, fill_value='extrapolate')
            interpolator.x_tran   = interp1d(pred.t + self.dt*np.arange(len(pred.x_tran)), pred.x_tran, kind='linear', assume_sorted=True, fill_value='extrapolate')
        else:
            _t = pred.t + self.dt*np.arange(len(pred.v_long))
            _v_long     = interp1d(_t, pred.v_long, kind='linear', assume_sorted=True)
            interpolator.v_long   = lambda t: _v_long(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            _t = pred.t + self.dt*np.arange(len(pred.v_tran))
            _v_tran     = interp1d(pred.t + self.dt*np.arange(len(pred.v_tran)), pred.v_tran, kind='linear', assume_sorted=True)
            interpolator.v_tran   = lambda t: _v_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            _t = pred.t + self.dt*np.arange(len(pred.psidot))
            _psidot     = interp1d(pred.t + self.dt*np.arange(len(pred.psidot)), pred.psidot, kind='linear', assume_sorted=True)
            interpolator.psidot   = lambda t: _psidot(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            _t = pred.t + self.dt*np.arange(len(pred.e_psi))
            _e_psi      = interp1d(pred.t + self.dt*np.arange(len(pred.e_psi)), pred.e_psi, kind='linear', assume_sorted=True)
            interpolator.e_psi    = lambda t: _e_psi(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            _t = pred.t + self.dt*np.arange(len(pred.s))
            _s          = interp1d(pred.t + self.dt*np.arange(len(pred.s)), pred.s, kind='linear', assume_sorted=True)
            interpolator.s        = lambda t: _s(np.maximum(np.minimum(t, _t[-1]), _t[0]))

            _t = pred.t + self.dt*np.arange(len(pred.x_tran))
            _x_tran     = interp1d(pred.t + self.dt*np.arange(len(pred.x_tran)), pred.x_tran, kind='linear', assume_sorted=True)
            interpolator.x_tran   = lambda t: _x_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))

        _t = pred.t + self.dt*np.arange(len(pred.u_a))
        _u_a        = interp1d(pred.t + self.dt*np.arange(len(pred.u_a)), pred.u_a, kind='linear', assume_sorted=True)
        interpolator.u_a      = lambda t: _u_a(np.maximum(np.minimum(t, _t[-1]), _t[0]))

        _t = pred.t + self.dt*np.arange(len(pred.u_steer))
        _u_steer    = interp1d(pred.t + self.dt*np.arange(len(pred.u_steer)), pred.u_steer, kind='linear', assume_sorted=True)
        interpolator.u_steer  = lambda t: _u_steer(np.maximum(np.minimum(t, _t[-1]), _t[0]))
        
        return interpolator
    
class CasadiDynamicBicycleCombined(CasadiDynamicsModel):
    '''
    Frenet frame of reference dynamic bicycle model - Pacejka model tire forces

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: DynamicBicycleConfig = DynamicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()
        self.get_tangent = self.track.get_tangent_angle_casadi_fn()

        self.n_q = 8
        self.n_u = 2

        self.L_f            = self.model_config.wheel_dist_front
        self.L_r            = self.model_config.wheel_dist_rear

        self.m              = self.model_config.mass
        self.I_z            = self.model_config.yaw_inertia
        self.g              = self.model_config.gravity

        self.c_dr           = self.model_config.drag_coefficient
        self.c_da           = self.model_config.damping_coefficient
        self.c_r            = self.model_config.rolling_resistance
        self.p_r            = self.model_config.rolling_resistance_exponent
        
        self.mu             = self.model_config.wheel_friction
        self.tire_model     = self.model_config.tire_model
        self.drive_wheels   = self.model_config.drive_wheels

        self.pacejka_Bf     = self.model_config.pacejka_b_front
        self.pacejka_Br     = self.model_config.pacejka_b_rear
        self.pacejka_Cf     = self.model_config.pacejka_c_front
        self.pacejka_Cr     = self.model_config.pacejka_c_rear
        self.pacejka_Df     = self.model_config.pacejka_d_front
        self.pacejka_Dr     = self.model_config.pacejka_d_rear

        self.linear_Bf      = self.model_config.linear_bf
        self.linear_Br      = self.model_config.linear_br

        self.simple_slip    = self.model_config.simple_slip

        self.use_mx         = self.model_config.use_mx
        if isinstance(self.track, CasadiBSplineTrack):
            self.use_mx = True

        if self.use_mx:
            sym = ca.MX.sym
        else:
            sym = ca.SX.sym
        
        # symbolic variables
        self.sym_x      = sym('x')
        self.sym_y      = sym('y')
        self.sym_vx     = sym('vx')
        self.sym_vy     = sym('vy')
        self.sym_psidot = sym('psidot')
        self.sym_epsi   = sym('epsi')
        self.sym_s      = sym('s')
        self.sym_xtran  = sym('xtran')
        self.sym_u_s    = sym('gamma')
        self.sym_u_a    = sym('a')

        self.sym_c = self.get_curvature(self.sym_s)
        self.sym_psi_t  = self.get_tangent(self.sym_s)

        # Slip angles and Pacejka tire forces
        if self.simple_slip:
            self.sym_alpha_f = -ca.atan2(self.sym_vy + self.L_f * self.sym_psidot, self.sym_vx) + self.sym_u_s
        else:
            self.sym_alpha_f = -ca.atan2(
                (self.sym_vy + self.L_f * self.sym_psidot) * ca.cos(self.sym_u_s) - self.sym_vx * ca.sin(self.sym_u_s),
                self.sym_vx * ca.cos(self.sym_u_s) + (self.sym_vy + self.L_f * self.sym_psidot) * ca.sin(self.sym_u_s))
        self.sym_alpha_r = -ca.atan2(self.sym_vy - self.L_r * self.sym_psidot, self.sym_vx)

        if self.tire_model == 'pacejka':
            self.sym_fyf = self.pacejka_Df * ca.sin(self.pacejka_Cf * ca.atan(self.pacejka_Bf * self.sym_alpha_f))
            self.sym_fyr = self.pacejka_Dr * ca.sin(self.pacejka_Cr * ca.atan(self.pacejka_Br * self.sym_alpha_r))
        elif self.tire_model == 'linear':
            self.sym_fyf = self.linear_Bf * self.m*self.g*self.L_r/(self.L_f+self.L_r) * self.sym_alpha_f
            self.sym_fyr = self.linear_Br * self.m*self.g*self.L_f/(self.L_f+self.L_r) * self.sym_alpha_r
        else:
            raise(ValueError("Tire model must be 'linear' or 'pacejka'"))

        # External forces
        F_ext = - self.c_da * self.sym_vx \
                - self.c_dr * self.sym_vx * self.ca_abs(self.sym_vx) \
                - self.c_r * self.ca_abs(self.sym_vx)**self.p_r * self.ca_sign(self.sym_vx)

        if self.drive_wheels == 'all':
            _ar = self.sym_u_a/2
            _af = self.sym_u_a/2
        elif self.drive_wheels == 'rear':
            _ar = self.sym_u_a
            _af = 0
        # instantaneous accelerations
        # self.sym_ax = self.sym_u_a + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        # self.sym_ay = (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_ax = _ar + _af*ca.cos(self.sym_u_s) + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        self.sym_ay = _af*ca.sin(self.sym_u_s) + (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m        
        self.sym_alphaz = (self.L_f * self.sym_fyf * ca.cos(self.sym_u_s) - self.L_r * self.sym_fyr) / self.I_z

        # time derivatives
        self.sym_dx         = self.sym_vx * ca.cos(self.sym_epsi + self.sym_psi_t) - self.sym_vy * ca.sin(self.sym_epsi + self.sym_psi_t)
        self.sym_dy         = self.sym_vy * ca.cos(self.sym_epsi + self.sym_psi_t) + self.sym_vx * ca.sin(self.sym_epsi + self.sym_psi_t)
        self.sym_dvx        = self.sym_ax + self.sym_psidot * self.sym_vy
        self.sym_dvy        = self.sym_ay - self.sym_psidot * self.sym_vx
        self.sym_dpsidot    = self.sym_alphaz
        self.sym_depsi      = self.sym_psidot - self.sym_c * (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds         = (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran     = self.sym_vx * ca.sin(self.sym_epsi) + self.sym_vy * ca.cos(self.sym_epsi)

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_x, self.sym_y, self.sym_vx, self.sym_vy, self.sym_psidot, self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dx, self.sym_dy, self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_depsi, self.sym_ds, self.sym_dxtran)

        # Auxilliary functions
        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_alpha = ca.Function('f_alpha', [self.sym_q, self.sym_u], [self.sym_alpha_f, self.sym_alpha_r])
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_alphaz], self.options('f_ang_a'))

        self.precompute_model()
        return

    def get_slip_angles(self, state: VehicleState) -> np.ndarray:
        q, u = self.state2qu(state)
        return np.array(self.f_alpha(q, u)).squeeze()

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.x.x, state.x.y, state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.x.x       = q[0]
        state.x.y       = q[1]
        state.v.v_long  = q[2]
        state.v.v_tran  = q[3]
        state.w.w_psi   = q[4]
        state.p.e_psi   = q[5]
        state.p.s       = q[6]
        state.p.x_tran  = q[7]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.x.x       = q[0]
            state.x.y       = q[1]
            state.v.v_long  = q[2]
            state.v.v_tran  = q[3]
            state.w.w_psi   = q[4]
            state.p.e_psi   = q[5]
            state.p.s       = q[6]
            state.p.x_tran  = q[7]
        if u is not None:
            state.u.u_a = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.x        = array.array('d', q[:, 0])
            prediction.y        = array.array('d', q[:, 1])
            prediction.v_long   = array.array('d', q[:, 2])
            prediction.v_tran   = array.array('d', q[:, 3])
            prediction.psidot   = array.array('d', q[:, 4])
            prediction.e_psi    = array.array('d', q[:, 5])
            prediction.s        = array.array('d', q[:, 6])
            prediction.x_tran   = array.array('d', q[:, 7])
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
        
        return prediction
    
    def qu2interpolator(self, t: float, q: np.ndarray = None, u: np.ndarray = None, extrapolate: bool = False):
        prediction = VehiclePrediction(t=t)
        if q is not None:
            _t = t + self.dt*np.arange(q.shape[0])

            if extrapolate:
                prediction.x          = interp1d(_t, q[:,0], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.y          = interp1d(_t, q[:,1], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.v_long     = interp1d(_t, q[:,2], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.v_tran     = interp1d(_t, q[:,3], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.psidot     = interp1d(_t, q[:,4], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.e_psi      = interp1d(_t, q[:,5], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.s          = interp1d(_t, q[:,6], kind='linear', assume_sorted=True, fill_value='extrapolate')
                prediction.x_tran     = interp1d(_t, q[:,7], kind='linear', assume_sorted=True, fill_value='extrapolate')
            else:
                _x          = interp1d(_t, q[:,0], kind='linear', assume_sorted=True)
                _y          = interp1d(_t, q[:,1], kind='linear', assume_sorted=True)
                _v_long     = interp1d(_t, q[:,2], kind='linear', assume_sorted=True)
                _v_tran     = interp1d(_t, q[:,3], kind='linear', assume_sorted=True)
                _psidot     = interp1d(_t, q[:,4], kind='linear', assume_sorted=True)
                _e_psi      = interp1d(_t, q[:,5], kind='linear', assume_sorted=True)
                _s          = interp1d(_t, q[:,6], kind='linear', assume_sorted=True)
                _x_tran     = interp1d(_t, q[:,7], kind='linear', assume_sorted=True)

                prediction.x        = lambda t: _x(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.y        = lambda t: _y(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.v_long   = lambda t: _v_long(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.v_tran   = lambda t: _v_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.psidot   = lambda t: _psidot(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.e_psi    = lambda t: _e_psi(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.s        = lambda t: _s(np.maximum(np.minimum(t, _t[-1]), _t[0]))
                prediction.x_tran   = lambda t: _x_tran(np.maximum(np.minimum(t, _t[-1]), _t[0]))

        if u is not None:
            _t = t + self.dt*np.arange(u.shape[0])
            _u_a        = interp1d(_t, u[:,0], kind='linear', assume_sorted=True)
            _u_steer    = interp1d(_t, u[:,1], kind='linear', assume_sorted=True)

            prediction.u_a      = lambda t: _u_a(np.maximum(np.minimum(t, _t[-1]), _t[0]))
            prediction.u_steer  = lambda t: _u_steer(np.maximum(np.minimum(t, _t[-1]), _t[0]))
        
        return prediction

class CasadiDynamicBicycleProgressAugmented(CasadiDynamicsModel):
    '''
    Global frame of reference dynamic bicycle model - Pacejka model tire forces

    Body frame velocities and global frame positions
    '''

    def __init__(self, t0: float,
                    model_config: DynamicBicycleConfig = DynamicBicycleConfig(), 
                    track=None,
                    track_tightening=0):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = False

        self.n_q = 7
        self.n_u = 3

        self.L_f            = self.model_config.wheel_dist_front
        self.L_r            = self.model_config.wheel_dist_rear

        self.m              = self.model_config.mass
        self.I_z            = self.model_config.yaw_inertia
        self.g              = self.model_config.gravity

        self.c_dr           = self.model_config.drag_coefficient
        self.c_da           = self.model_config.damping_coefficient
        self.c_r            = self.model_config.rolling_resistance
        self.p_r            = self.model_config.rolling_resistance_exponent

        self.mu             = self.model_config.wheel_friction
        self.tire_model     = self.model_config.tire_model
        self.drive_wheels   = self.model_config.drive_wheels

        self.pacejka_Bf     = self.model_config.pacejka_b_front
        self.pacejka_Br     = self.model_config.pacejka_b_rear
        self.pacejka_Cf     = self.model_config.pacejka_c_front
        self.pacejka_Cr     = self.model_config.pacejka_c_rear
        self.pacejka_Df     = self.model_config.pacejka_d_front
        self.pacejka_Dr     = self.model_config.pacejka_d_rear

        self.linear_Bf      = self.model_config.linear_bf
        self.linear_Br      = self.model_config.linear_br

        self.simple_slip    = self.model_config.simple_slip

        self.code_gen       = self.model_config.code_gen
        self.opt_flag       = self.model_config.opt_flag

        self.use_mx         = self.model_config.use_mx
        
        # symbolic variables
        self.sym_vx         = ca.SX.sym('vx')  # body fram vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_vy         = ca.SX.sym('vy')
        self.sym_psidot     = ca.SX.sym('psidot')
        self.sym_x          = ca.SX.sym('x')
        self.sym_y          = ca.SX.sym('y')
        self.sym_psi        = ca.SX.sym('psi')
        self.sym_s          = ca.SX.sym('steer')
        self.sym_u_s        = ca.SX.sym('accel')
        self.sym_u_a        = ca.SX.sym('a')
        self.sym_u_ds       = ca.SX.sym('ds')

        # Slip angles and Pacejka tire forces
        if self.simple_slip:
            self.sym_alpha_f = -ca.atan2(self.sym_vy + self.L_f * self.sym_psidot, self.sym_vx) + self.sym_u_s
        else:
            self.sym_alpha_f = -ca.atan2((self.sym_vy + self.L_f * self.sym_psidot) * ca.cos(self.sym_u_s) - self.sym_vx * ca.sin(self.sym_u_s),
                                         self.sym_vx * ca.cos(self.sym_u_s) + (self.sym_vy + self.L_f * self.sym_psidot) * ca.sin(self.sym_u_s))
        self.sym_alpha_r = -ca.atan2(self.sym_vy - self.L_r * self.sym_psidot, self.sym_vx)

        if self.tire_model == 'pacejka':
            self.sym_fyf = self.pacejka_Df * ca.sin(self.pacejka_Cf * ca.atan(self.pacejka_Bf * self.sym_alpha_f))
            self.sym_fyr = self.pacejka_Dr * ca.sin(self.pacejka_Cr * ca.atan(self.pacejka_Br * self.sym_alpha_r))
        elif self.tire_model == 'linear':
            self.sym_fyf = self.linear_Bf * self.m*self.g*self.L_r/(self.L_f+self.L_r) * self.sym_alpha_f
            self.sym_fyr = self.linear_Br * self.m*self.g*self.L_f/(self.L_f+self.L_r) * self.sym_alpha_r
        else:
            raise(ValueError("Tire model must be 'linear' or 'pacejka'"))

        # External forces
        F_ext = - self.c_da * self.sym_vx \
                - self.c_dr * self.sym_vx * self.ca_abs(self.sym_vx) \
                - self.c_r * self.ca_abs(self.sym_vx)**self.p_r * self.ca_sign(self.sym_vx)

        if self.drive_wheels == 'all':
            _ar = self.sym_u_a/2
            _af = self.sym_u_a/2
        elif self.drive_wheels == 'rear':
            _ar = self.sym_u_a
            _af = 0
        # instantaneous accelerations
        # self.sym_ax = self.sym_u_a + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        # self.sym_ay = (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_ax = _ar + _af*ca.cos(self.sym_u_s) + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        self.sym_ay = _af*ca.sin(self.sym_u_s) + (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m        
        self.sym_alphaz = (self.L_f * self.sym_fyf * ca.cos(self.sym_u_s) - self.L_r * self.sym_fyr) / self.I_z

        # time derivatives
        self.sym_dvx        = self.sym_ax + self.sym_psidot * self.sym_vy
        self.sym_dvy        = self.sym_ay - self.sym_psidot * self.sym_vx
        self.sym_dpsidot    = self.sym_alphaz
        self.sym_dx         = self.sym_vx * ca.cos(self.sym_psi) - self.sym_vy * ca.sin(self.sym_psi)
        self.sym_dy         = self.sym_vy * ca.cos(self.sym_psi) + self.sym_vx * ca.sin(self.sym_psi)
        self.sym_dpsi       = self.sym_psidot
        self.sym_ds         = self.sym_u_ds

        # state and state derivative functions
        self.sym_q  = ca.vertcat(self.sym_vx, self.sym_vy, self.sym_psidot, self.sym_x, self.sym_y, self.sym_psi, self.sym_s)
        self.sym_u  = ca.vertcat(self.sym_u_a, self.sym_u_s, self.sym_u_ds)
        self.sym_dq = ca.vertcat(self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_dx, self.sym_dy, self.sym_dpsi, self.sym_ds)
        
        # Auxilliary functions
        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay, 0], self.options('f_a'))
        self.f_alpha = ca.Function('f_alpha', [self.sym_q, self.sym_u], [self.sym_alpha_f, self.sym_alpha_r])
        self.f_ang_a = ca.Function('f_ang_a', [self.sym_q, self.sym_u], [0, 0, self.sym_alphaz], self.options('f_ang_a'))

        self.precompute_model()

        if isinstance(self.track, RadiusArclengthTrack):
            # Compute spline approximation of track
            S = np.linspace(0, self.track.track_length, 100)
            X, Y, Xi, Yi, Xo, Yo = [], [], [], [], [], []
            for s in S:
                # Centerline
                x, y, _ = self.track.local_to_global((s, 0, 0))
                X.append(x)
                Y.append(y)
                # Inside boundary
                xi, yi, _ = self.track.local_to_global((s, self.track.half_width-track_tightening, 0))
                Xi.append(xi)
                Yi.append(yi)
                # Outside boundary
                xo, yo, _ = self.track.local_to_global((s, -(self.track.half_width-track_tightening), 0))
                Xo.append(xo)
                Yo.append(yo)
            self.x_s = ca.interpolant('x_s', 'bspline', [S], X)
            self.y_s = ca.interpolant('y_s', 'bspline', [S], Y)
            self.xi_s = ca.interpolant('xi_s', 'bspline', [S], Xi)
            self.yi_s = ca.interpolant('yi_s', 'bspline', [S], Yi)
            self.xo_s = ca.interpolant('xo_s', 'bspline', [S], Xo)
            self.yo_s = ca.interpolant('yo_s', 'bspline', [S], Yo)

            # Compute derivatives of track
            s_sym = ca.MX.sym('s', 1)
            self.dxds = ca.Function('dxds', [s_sym], [ca.jacobian(self.x_s(s_sym), s_sym)])
            self.dyds = ca.Function('dyds', [s_sym], [ca.jacobian(self.y_s(s_sym), s_sym)])
        elif isinstance(self.track, CasadiBSplineTrack):
            self.x_s = self.track.x
            self.y_s = self.track.y
            self.xi_s = self.track.xi
            self.yi_s = self.track.yi
            self.xo_s = self.track.xo
            self.yo_s = self.track.yo

            self.dxds = self.track.dx
            self.dyds = self.track.dy
        else:
            raise(ValueError(f'Track type {type(self.track)} not supported'))

        self.pos_idx = [3, 4]
        return

    def get_contouring_lag_costs(self, contouring_cost, lag_cost):
        sym_q = ca.MX.sym('q', self.n_q)
        L = self.track.track_length
        # Contouring and lag errors and their gradients
        # s_mod = ca.fmod(sym_q[-1], L)
        s_mod = ca.fmod(ca.fmod(sym_q[-1], L) + L, L)
        # Reference interpolation variable must be in range [-1, 1] (outside, inside)
        z_sym = ca.MX.sym('z', 1)
        t = ca.atan2(self.dyds(s_mod), self.dxds(s_mod))
        x_int = self.xo_s(s_mod) + (z_sym+1)/2*(self.xi_s(s_mod)-self.xo_s(s_mod))
        y_int = self.yo_s(s_mod) + (z_sym+1)/2*(self.yi_s(s_mod)-self.yo_s(s_mod))
        ec =  ca.sin(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.cos(t)*(sym_q[self.pos_idx[1]]-y_int)
        el = -ca.cos(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.sin(t)*(sym_q[self.pos_idx[1]]-y_int)
        f_e = ca.Function('ec', [sym_q, z_sym], [ca.vertcat(contouring_cost*ec, lag_cost*el)])
        return f_e

    def get_contouring_lag_costs_quad_approx(self, contouring_cost, lag_cost):
        sym_q = ca.MX.sym('q', self.n_q)
        L = self.track.track_length
        # Contouring and lag errors and their gradients
        # s_mod = ca.fmod(sym_q[-1], L)
        s_mod = ca.fmod(ca.fmod(sym_q[-1], L) + L, L)
        # Reference interpolation variable must be in range [-1, 1] (outside, inside)
        z_sym = ca.MX.sym('z', 1)
        t = ca.atan2(self.dyds(s_mod), self.dxds(s_mod))
        x_int = self.xo_s(s_mod) + (z_sym+1)/2*(self.xi_s(s_mod)-self.xo_s(s_mod))
        y_int = self.yo_s(s_mod) + (z_sym+1)/2*(self.yi_s(s_mod)-self.yo_s(s_mod))
        ec =  ca.sin(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.cos(t)*(sym_q[self.pos_idx[1]]-y_int)
        el = -ca.cos(t)*(sym_q[self.pos_idx[0]]-x_int) - ca.sin(t)*(sym_q[self.pos_idx[1]]-y_int)
        
        e = ca.vertcat(ec, el)
        Dx_e = ca.jacobian(e, sym_q)

        P_cl = ca.diag(ca.vertcat(contouring_cost, lag_cost))

        # The approximation is (1/2) q.T @ Q_e(q_bar) @ q + q_e(q_bar).T @ q
        Q_e = Dx_e.T @ P_cl @ Dx_e
        q_e = Dx_e.T @ P_cl @ e - Q_e @ sym_q
        
        options = dict(jit=self.code_gen, jit_name='contouring_lag_approx', compiler='shell', jit_options=dict(compiler='gcc', flags=[f'-{self.opt_flag}'], verbose=False))
        f_cl = ca.Function('contouring_lag_approx', [sym_q, z_sym], [Q_e, q_e], options)

        return f_cl

    def get_track_boundary_constraint_lin_approx(self):
        sym_q = ca.MX.sym('q', self.n_q)
        L = self.track.track_length
        # s_mod = ca.fmod(sym_q[-1], L)
        s_mod = ca.fmod(ca.fmod(sym_q[-1], L) + L, L)

        # Linear approximation of track boundary constraints
        xi, yi = self.xi_s(s_mod), self.yi_s(s_mod)
        xo, yo = self.xo_s(s_mod), self.yo_s(s_mod)
        n, d = -(xo - xi), yo - yi

        # The approximation is G @ x + g <= 0
        G = ca.MX.sym('G', ca.Sparsity(2, self.n_q))
        G[0,self.pos_idx[0]], G[0,self.pos_idx[1]] = n, -d
        G[1,self.pos_idx[0]], G[1,self.pos_idx[1]] = -n, d
        g = ca.vertcat(-ca.fmax(n*xi-d*yi, n*xo-d*yo), ca.fmin(n*xi-d*yi, n*xo-d*yo))

        options = dict(jit=self.code_gen, jit_name='track_boundary_approx', compiler='shell', jit_options=dict(compiler='gcc', flags=[f'-{self.opt_flag}'], verbose=False))
        f_tb = ca.Function('G', [sym_q], [G, g], options)

        return f_tb

    def get_arcspeed_cost(self, magnitude_weight, performance_weight):
        sym_u = ca.SX.sym('u', self.n_u)
        arcspeed_cost = (1/2)*magnitude_weight*sym_u[-1]**2 - performance_weight*sym_u[-1]
        f_u = ca.Function('u', [sym_u], [arcspeed_cost])
        return f_u

    def get_slip_angles(self, state: VehicleState) -> np.ndarray:
        q, u = self.state2qu(state)
        return np.array(self.f_alpha(q, u)).squeeze()

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi, state.p.s])
        u = np.array([state.u.u_a, state.u.u_steer, state.u.u_ds])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi, state.p.s])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer, input.u_ds])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long  = q[0]
        state.v.v_tran  = q[1]
        state.w.w_psi   = q[2]
        state.x.x       = q[3]
        state.x.y       = q[4]
        state.e.psi     = q[5]
        state.p.s       = q[6]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        input.u_ds      = u[2]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long  = q[0]
            state.v.v_tran  = q[1]
            state.w.w_psi   = q[2]
            state.x.x       = q[3]
            state.x.y       = q[4]
            state.e.psi     = q[5]
            state.p.s       = q[6]
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
            state.u.u_ds    = u[2]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.v_long   = array.array('d', q[:, 0])
            prediction.v_tran   = array.array('d', q[:, 1])
            prediction.psidot   = array.array('d', q[:, 2])
            prediction.x        = array.array('d', q[:, 3])
            prediction.y        = array.array('d', q[:, 4])
            prediction.psi      = array.array('d', q[:, 5])
            prediction.s        = array.array('d', q[:, 6])
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])
            prediction.u_ds     = array.array('d', u[:, 2])

        return prediction

class CasadiDecoupledMultiAgentDynamicsModel(CasadiDynamicsModel):
    def __init__(self, t0: float, dynamics_models: List[CasadiDynamicsModel], model_config: MultiAgentModelConfig = MultiAgentModelConfig()):
        super().__init__(t0, model_config, track=None)

        self.dynamics_models = dynamics_models
        self.n_a = len(self.dynamics_models) # Number of agents

        self.use_mx = model_config.use_mx
        for m in self.dynamics_models:
            if m.use_mx:
                self.use_mx = True
                break
        
        self.n_q, self.n_u = 0, 0
        for i in range(self.n_a):
            self.n_q += self.dynamics_models[i].n_q # Joint state dimension
            self.n_u += self.dynamics_models[i].n_u # Joint input dimension

        if self.use_mx:
            # Define symbolic variables for joint vectors
            self.sym_q = ca.MX.sym('q', self.n_q) # State
            self.sym_dq = ca.MX.sym('dq', self.n_q) # State derivative
            self.sym_u = ca.MX.sym('u', self.n_u) # Input
            # Split into agent vectors
            sym_q, sym_u, = [], []
            q_start, u_start = 0, 0
            for i in range(self.n_a):
                sym_q.append(ca.vertcat(*ca.vertsplit(self.sym_q)[q_start:q_start+self.dynamics_models[i].n_q]))
                sym_u.append(ca.vertcat(*ca.vertsplit(self.sym_u)[u_start:u_start+self.dynamics_models[i].n_u]))
                q_start += self.dynamics_models[i].n_q
                u_start += self.dynamics_models[i].n_u
        else:
            # Define symbolic variables for each agent
            sym_q = [ca.SX.sym('q_%i' % i, self.dynamics_models[i].n_q) for i in range(self.n_a)]
            sym_u = [ca.SX.sym('u_%i' % i, self.dynamics_models[i].n_u) for i in range(self.n_a)]
            # Concatenate into joint vector (we do this because slicing is inefficient in CasADi)
            self.sym_q = ca.vertcat(*sym_q)
            self.sym_u = ca.vertcat(*sym_u)

        sym_dq = []
        # Assemble vehicle dynamics input arguments for each agent
        for i, dyn_mdl in enumerate(self.dynamics_models):
            sym_dq.append(dyn_mdl.fc(sym_q[i], sym_u[i]))

        self.sym_dq = ca.vertcat(*sym_dq)

        self.precompute_model()
    
        return

    def step(self, vehicle_states: List[VehicleState],
            method: str = 'RK45'):
        '''
        steps noise-free model forward one time step (self.dt) using numerical integration
        '''
        q, u = self.state2qu(vehicle_states)

        f = lambda t, qs: (self.fc(qs, u)).toarray().squeeze()

        t = vehicle_states[0].t - self.t0
        tf = t + self.dt
        q_n = solve_ivp(f, [t,tf], q, method = method).y[:,-1]

        self.qu2state(vehicle_states, q_n, u)
        for i in range(self.n_a):
            vehicle_states[i].t = tf + self.t0
            vehicle_states[i].a.a_long, vehicle_states[i].a.a_tran, vehicle_states[i].a.a_long, vehicle_states[i].a.a_n = self.dynamics_models[i].f_a(q_n, u)
            vehicle_states[i].aa.a_phi, vehicle_states[i].aa.a_theta, vehicle_states[i].aa.a_psi = self.dynamics_models[i].f_ang_a(q_n, u)

            if self.curvature_model:
                self.track.local_to_global_typed(vehicle_states[i])
            else:
                self.track.global_to_local_typed(vehicle_states[i])
        return

    def state2qu(self, vehicle_states: List[VehicleState]) -> Tuple[np.ndarray, np.ndarray]:
        q_joint = []
        u_joint = []

        for i in range(self.n_a):
            q, u = self.dynamics_models[i].state2qu(vehicle_states[i])
            q_joint.append(q)
            u_joint.append(u)

        return np.concatenate(q_joint), np.concatenate(u_joint)

    def state2q(self, vehicle_states: List[VehicleState]) -> np.ndarray:
        q_joint = []

        for i in range(self.n_a):
            q = self.dynamics_models[i].state2q(vehicle_states[i])
            q_joint.append(q)

        return np.concatenate(q_joint)

    def qu2state(self, vehicle_states: List[VehicleState], q_joint: np.ndarray = None, u_joint: np.ndarray = None):
        if vehicle_states is None:
            vehicle_states = [VehicleState() for _ in range(self.n_a)]
        q_idx_start = 0
        u_idx_start = 0
        for i in range(self.n_a):
            if q_joint is not None:
                if u_joint is not None:
                    self.dynamics_models[i].qu2state(vehicle_states[i], q_joint[q_idx_start:q_idx_start+self.dynamics_models[i].n_q], u_joint[u_idx_start:u_idx_start+self.dynamics_models[i].n_u])
                    u_idx_start += self.dynamics_models[i].n_u
                else:
                    self.dynamics_models[i].qu2state(vehicle_states[i], q_joint[q_idx_start:q_idx_start+self.dynamics_models[i].n_q], None)
                q_idx_start += self.dynamics_models[i].n_q
            else:
                if u_joint is not None:
                    self.dynamics_models[i].qu2state(vehicle_states[i], None, u_joint[u_idx_start:u_idx_start+self.dynamics_models[i].n_u])
                    u_idx_start += self.dynamics_models[i].n_u
        return vehicle_states

    def qu2prediction(self, state_predictions: List[VehiclePrediction], q_pred: np.ndarray = None, u_pred: np.ndarray = None):
        if state_predictions is None:
            state_predictions = [VehiclePrediction() for _ in range(self.n_a)]
        q_idx_start = 0
        u_idx_start = 0
        for i in range(self.n_a):
            n_q = self.dynamics_models[i].n_q
            n_u = self.dynamics_models[i].n_u
            if q_pred is not None:
                if u_pred is not None:
                    self.dynamics_models[i].qu2prediction(state_predictions[i], q_pred[:,q_idx_start:q_idx_start+n_q], u_pred[:,u_idx_start:u_idx_start+n_u])
                    u_idx_start += n_u
                else:
                    self.dynamics_models[i].qu2prediction(state_predictions[i], q_pred[:,q_idx_start:q_idx_start+n_q], None)
                q_idx_start += n_q
            else:
                if u_pred is not None:
                    self.dynamics_models[i].qu2prediction(state_predictions[i], None, u_pred[:,u_idx_start:u_idx_start+n_u])
                    u_idx_start += n_u
        return state_predictions

    def input2u(self, vehicle_inputs: List[VehicleActuation]) -> np.ndarray:
        u_joint = []
        for i in range(self.n_a):
            u = self.dynamics_models[i].input2u(vehicle_inputs[i])
            u_joint.append(u)

        return np.concatenate(u_joint)

    def q2state(self, vehicle_states: List[VehicleState], q_joint: np.ndarray):
        if vehicle_states is None:
            vehicle_states = [VehicleState() for _ in range(self.n_a)]
        q_idx_start = 0
        for i in range(self.n_a):
            self.dynamics_models[i].q2state(vehicle_states[i], q_joint[q_idx_start:q_idx_start+self.dynamics_models[i].n_q])
            q_idx_start += self.dynamics_models[i].n_q
        return vehicle_states

def get_dynamics_model(t_start: float, model_config: DynamicsConfig, track=None) -> CasadiDynamicsModel:
    '''
    Helper function for getting a vehicle model class from a text string
    Should be used anywhere vehicle models may be changed by configuration
    '''
    if model_config.model_name == 'dynamic_bicycle':
        return CasadiDynamicBicycle(t_start, model_config, track=track)
    elif model_config.model_name == 'dynamic_bicycle_cl':
        return CasadiDynamicCLBicycle(t_start, model_config, track=track)
    elif model_config.model_name == 'kinematic_bicycle':
        return CasadiKinematicBicycle(t_start, model_config, track=track)
    elif model_config.model_name == 'kinematic_bicycle_cl':
        return CasadiKinematicCLBicycle(t_start, model_config, track=track)
    elif model_config.model_name == 'kinematic_bicycle_combined':
        return CasadiKinematicBicycleCombined(t_start, model_config, track=track)
    elif model_config.model_name == 'kinematic_unicycle_combined':
        return CasadiKinematicUnicycleCombined(t_start, model_config, track=track)
    elif model_config.model_name == 'kinematic_unicycle':
        return CasadiKinematicUnicycle(t_start, model_config, track=track)
    else:
        raise ValueError('Unrecognized vehicle model name: %s' % model_config.model_name)
