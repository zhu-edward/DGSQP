#!/usr/bin python3

import numpy as np
import time

from typing import Tuple

from DGSQP.types import VehicleState

from DGSQP.solvers.abstract_solver import AbstractSolver
from DGSQP.solvers.solver_types import PIDParams

class PID():
    '''
    Base class for PID controller
    Meant to be packaged for use in actual controller (eg. ones that operate directly on vehicle state) since a PID controller by itself is not sufficient for vehicle control
    See PIDLaneFollower for a PID controller that is an actual controller
    '''
    def __init__(self, params: PIDParams = PIDParams()):
        self.dt             = params.dt

        self.Kp             = params.Kp             # proportional gain
        self.Ki             = params.Ki             # integral gain
        self.Kd             = params.Kd             # derivative gain

        # Integral action and control action saturation limits
        self.int_e_max      = params.int_e_max
        self.int_e_min      = params.int_e_min
        self.u_max          = params.u_max
        self.u_min          = params.u_min
        self.du_max         = params.du_max
        self.du_min         = params.du_min

        # Add random noise
        self.noise          = params.noise
        self.noise_min      = params.noise_min
        self.noise_max      = params.noise_max

        # Add periodic disturbance
        self.periodic_disturbance = params.periodic_disturbance
        self.disturbance_amplitude = params.disturbance_amplitude
        self.disturbance_period = params.disturbance_period

        self.x_ref          = 0
        self.u_ref          = 0

        self.e              = 0             # error
        self.de             = 0             # finite time error difference
        self.ei             = 0             # accumulated error

        self.time_execution = True
        self.t0 = None

        self.initialized = False

    def initialize(self,
                    x_ref: float = 0,
                    u_ref: float = 0,
                    de: float = 0,
                    ei: float = 0,
                    time_execution: bool = False):
        self.de = de
        self.ei = ei

        self.x_ref = x_ref         # reference point
        self.u_ref = u_ref         # control signal offset

        self.time_execution = time_execution
        self.t0 = time.time()
        self.u_prev = None
        self.initialized = True

    def solve(self, x: float,
                u_prev: float = None) -> Tuple[float, dict]:
        if not self.initialized:
            raise(RuntimeError('PID controller is not initialized, run PID.initialize() before calling PID.solve()'))

        if self.u_prev is None and u_prev is None: u_prev = 0
        elif u_prev is None: u_prev = self.u_prev

        if self.time_execution:
            t_s = time.time()

        info = {'success' : True}

        # Compute error terms
        e_t = x - self.x_ref
        de_t = (e_t - self.e)/self.dt
        ei_t = self.ei + e_t*self.dt

        # Anti-windup
        if ei_t > self.int_e_max:
            ei_t = self.int_e_max
        elif ei_t < self.int_e_min:
            ei_t = self.int_e_min

        # Compute control action terms
        P_val  = self.Kp * e_t
        I_val  = self.Ki * ei_t
        D_val  = self.Kd * de_t

        u = -(P_val + I_val + D_val) + self.u_ref
        if self.noise:
            w = np.random.uniform(low=self.noise_min, high=self.noise_max)
            u += w
        if self.periodic_disturbance:
            t = time.time() - self.t0
            w = self.disturbance_amplitude*np.sin(2*np.pi*t/self.disturbance_period)
            u += w

        # Compute change in control action from previous timestep
        du = u - u_prev

        # Saturate change in control action
        if self.du_max is not None:
            du = self._saturate_rel_high(du)
        if self.du_min is not None:
            du = self._saturate_rel_low(du)

        u = du + u_prev

        # Saturate absolute control action
        if self.u_max is not None:
            u = self._saturate_abs_high(u)
        if self.u_min is not None:
            u = self._saturate_abs_low(u)

        # Update error terms
        self.e  = e_t
        self.de = de_t
        self.ei = ei_t

        if self.time_execution:
            info['solve_time'] = time.time() - t_s

        self.u_prev = u
        return u, info

    def set_x_ref(self, x: float, x_ref: float):
        self.x_ref = x_ref
        # reset error integrator
        self.ei = 0
        # reset error, otherwise de/dt will skyrocket
        self.e = x - x_ref

    def set_u_ref(self, u_ref: float):
        self.u_ref = u_ref

    def clear_errors(self):
        self.ei = 0
        self.de = 0

    def set_params(self, params:  PIDParams):
        self.dt             = params.dt

        self.Kp             = params.Kp             # proportional gain
        self.Ki             = params.Ki             # integral gain
        self.Kd             = params.Kd             # derivative gain

        # Integral action and control action saturation limits
        self.int_e_max      = params.int_e_max
        self.int_e_min      = params.int_e_min
        self.u_max          = params.u_max
        self.u_min          = params.u_min
        self.du_max         = params.du_max
        self.du_min         = params.du_min

    def get_refs(self) -> Tuple[float, float]:
        return (self.x_ref, self.u_ref)

    def get_errors(self) -> Tuple[float, float, float]:
        return (self.e, self.de, self.ei)

    def _saturate_abs_high(self, u: float) -> float:
        return np.minimum(u, self.u_max)

    def _saturate_abs_low(self, u: float) -> float:
        return np.maximum(u, self.u_min)

    def _saturate_rel_high(self, du: float) -> float:
        return np.minimum(du, self.du_max)

    def _saturate_rel_low(self, du: float) -> float:
        return np.maximum(du, self.du_min)

class PIDLaneFollower(AbstractSolver):
    '''
    Class for PID throttle and steering control of a vehicle
    Incorporates separate PID controllers for maintaining a constant speed and a constant lane offset

    target speed: v_ref
    target lane offset_ x_ref


    '''
    def __init__(self, v_ref: float, x_ref: float, dt: float,
                steer_pid_params: PIDParams = None,
                speed_pid_params: PIDParams = None):
        if steer_pid_params is None:
            steer_pid_params = PIDParams()
            steer_pid_params.dt = dt
            steer_pid_params.default_steer_params()
        if speed_pid_params is None:
            speed_pid_params = PIDParams()
            speed_pid_params.dt = dt
            speed_pid_params.default_speed_params()  # these may use dt so it is updated first

        self.dt = dt
        steer_pid_params.dt = dt
        speed_pid_params.dt = dt

        self.steer_pid = PID(steer_pid_params)
        self.speed_pid = PID(speed_pid_params)

        self.v_ref = v_ref
        self.x_ref = x_ref
        self.speed_pid.initialize(self.v_ref)
        self.steer_pid.initialize(0)

        self.requires_env_state = False
        return

    def initialize(self, **args):
        return

    def solve(self, **args):
        raise NotImplementedError('PID Lane follower does not implement a solver of its own')
        return

    def step(self, vehicle_state: VehicleState, env_state = None):
        v = np.sqrt(vehicle_state.v.v_long**2 + vehicle_state.v.v_tran**2)

        vehicle_state.u.u_a, _ = self.speed_pid.solve(v)
        # Weighting factor: alpha*x_trans + beta*psi_diff
        alpha = 5.0
        beta = 1.0
        vehicle_state.u.u_steer, _ = self.steer_pid.solve(alpha*(vehicle_state.p.x_tran - self.x_ref) + beta*vehicle_state.p.e_psi)
        return

# Test script to ensure controller object is functioning properly
if __name__ == "__main__":
    import pdb

    params = PIDParams(dt=0.1, Kp=3.7, Ki=7, Kd=0.5)
    x_ref = 5
    pid = PID(params)
    # pdb.set_trace()
    pid.initialize(x_ref=x_ref)
    # pdb.set_trace()

    print('Controller instantiated successfully')
