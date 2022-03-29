#!/usr/bin python3

from dataclasses import dataclass, field
import numpy as np

from DGSQP.types import PythonMsg, VehicleState

@dataclass
class ControllerConfig(PythonMsg):
    dt: float = field(default=0.1)

@dataclass
class PIDParams(ControllerConfig):
    Kp: float = field(default=2.0)
    Ki: float = field(default=0.0)
    Kd: float = field(default=0.0)

    int_e_max: float = field(default=100)
    int_e_min: float = field(default=-100)
    u_max: float = field(default=None)
    u_min: float = field(default=None)
    du_max: float = field(default=None)
    du_min: float = field(default=None)

    noise: bool = field(default=False)
    noise_max: float = field(default=0.1)
    noise_min: float = field(default=-0.1)

    periodic_disturbance: bool = field(default=False)
    disturbance_amplitude: float = field(default=0.1)
    disturbance_period: float = field(default=1.0)

    def default_speed_params(self):
        self.Kp = 1
        self.Ki = 0
        self.Kd = 0
        self.u_min = -2
        self.u_max = 2
        self.du_min = -10 * self.dt
        self.du_max =  10 * self.dt
        self.noise = False
        return

    def default_steer_params(self):
        self.Kp = 1
        self.Ki = 0.0005 / self.dt
        self.Kd = 0
        self.u_min = -0.35
        self.u_max = 0.35
        self.du_min = -4 * self.dt
        self.du_max = 4 * self.dt
        self.noise = False
        return

@dataclass
class ALGAMESParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    rho: float              = field(default=1.0) # Lagrangian regularization
    gamma: float            = field(default=10.0) # rho update schedule
    rho_max: float          = field(default=1e7)
    lam_max: float          = field(default=1e7)

    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param

    q_reg: float            = field(default=1e-2) # Jacobian regularization
    u_reg: float            = field(default=1e-2) # Jacobian regularization
    line_search_tol: float  = field(default=1e-6)
    newton_step_tol: float  = field(default=1e-6) # Newton step size
    ineq_tol: float         = field(default=1e-3) # Inequality constraint violation
    eq_tol: float           = field(default=1e-3) # Equality constraint violation
    opt_tol: float          = field(default=1e-3) # Optimality violation

    outer_iters: int        = field(default=50)
    line_search_iters: int  = field(default=50)
    newton_iters: int       = field(default=50)

    verbose: bool           = field(default=False)
    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='ALGAMES')
    solver_dir: str         = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class DGSQPParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param

    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    reg: float              = field(default=1e-3)
    line_search_iters: int  = field(default=50)
    nonmono_ls: bool        = field(default=False)
    sqp_iters: int          = field(default=50)
    conv_approx: bool       = field(default=True)
    merit_function: str     = field(default='stat_l1')

    verbose: bool           = field(default=False)
    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='DGSQP')
    solver_dir: str         = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class IBRParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    use_ps: bool            = field(default=True)
    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    line_search_iters: int  = field(default=50)
    ibr_iters: int          = field(default=50)

    verbose: bool           = field(default=False)
    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='IBR')
    solver_dir: str         = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

if __name__ == "__main__":
    pass
