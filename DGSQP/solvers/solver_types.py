#!/usr/bin python3

from dataclasses import dataclass, field

from DGSQP.types import PythonMsg

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

    u_ref: float = field(default=0.0)
    x_ref: float = field(default=0.0)

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

    dynamics_hessians: bool = field(default=False)

    outer_iters: int        = field(default=50)
    line_search_iters: int  = field(default=50)
    newton_iters: int       = field(default=50)

    verbose: bool           = field(default=False)
    solver_name: str        = field(default='ALGAMES')

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
    merit_function: str     = field(default='stat_l1')

    verbose: bool           = field(default=False)
    save_iter_data: bool    = field(default=True)

    solver_name: str        = field(default='DGSQP')
    time_limit: float       = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class DGSQPV2Params(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    beta: float             = field(default=0.25) # Line search param
    tau: float              = field(default=0.5) # Line search param

    p_tol: float            = field(default=1e-4)
    d_tol: float            = field(default=1e-4)

    reg: float              = field(default=1e2)
    reg_decay: float        = field(default=0.95)
    line_search_iters: int  = field(default=50)
    nms: bool               = field(default=True)
    nms_frequency: int      = field(default=5)
    nms_memory_size: int    = field(default=3)
    sqp_iters: int          = field(default=500)
    merit_function: str     = field(default='stat_l1')
    merit_parameter: float  = field(default=None)
    merit_decrease: float   = field(default=0.01)
    merit_decrease_condition: str = field(default='armijo') # 'max'
    approximation_eval: str = field(default='always') # 'once'
    delta_decay: float      = field(default=0.95)

    verbose: bool           = field(default=False)
    save_iter_data: bool    = field(default=False)
    save_qp_data: bool      = field(default=False)
    time_limit: float       = field(default=None)

    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='DGSQP')
    solver_dir: str         = field(default=None)
    so_name: str            = field(default=None)
    qp_interface: str       = field(default='casadi')
    qp_solver: str          = field(default='osqp')
    hessian_approximation: str = field(default='none')

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    save_plot: bool         = field(default=False)
    show_ts: bool           = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class IBRParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    use_ps: bool            = field(default=True)
    p_tol: float            = field(default=1e-3)
    d_tol: float            = field(default=1e-3)

    line_search_iters: int  = field(default=50)
    ibr_iters: int          = field(default=50)

    linear_solver: str      = field(default='mumps')

    verbose: bool           = field(default=False)
    code_gen: bool          = field(default=False)
    jit: bool               = field(default=False)
    opt_flag: str           = field(default='O0')
    enable_jacobians: bool  = field(default=True)
    solver_name: str        = field(default='PSIBR')
    solver_dir: str         = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)
    local_pos: bool         = field(default=False)

@dataclass
class CALTVMPCParams(ControllerConfig):
    N: int                              = field(default=10) # horizon length
    
    # Code gen options
    verbose: bool                       = field(default=False)
    code_gen: bool                      = field(default=False)
    jit: bool                           = field(default=False)
    opt_flag: str                       = field(default='O0')
    solver_name: str                    = field(default='LTV_MPC')
    solver_dir: str                     = field(default=None)
    debug_plot: bool                    = field(default=False)

    soft_state_bound_idxs: list         = field(default=None)
    soft_state_bound_quad: list         = field(default=None)
    soft_state_bound_lin: list          = field(default=None)

    soft_constraint_idxs: list         = field(default=None)
    soft_constraint_quad: list         = field(default=None)
    soft_constraint_lin: list          = field(default=None)

    wrapped_state_idxs: list            = field(default=None)
    wrapped_state_periods: list         = field(default=None)

    reg: float                          = field(default=0)
    state_scaling: list                 = field(default=None)
    input_scaling: list                 = field(default=None)
    damping: float                      = field(default=0.75)
    qp_iters: int                       = field(default=2)
    qp_interface: str                   = field(default='casadi')

    delay: list                         = field(default=None)

@dataclass
class PATHMCPParams(ControllerConfig):
    N: int                  = field(default=10) # Horizon length

    p_tol: float            = field(default=1e-3)

    nms: bool               = field(default=True)
    nms_frequency: int      = field(default=5)
    nms_memory_size: int    = field(default=3)
    
    outer_iters: int        = field(default=50)
    inner_iters: int        = field(default=50)
    
    solver_name: str        = field(default='PATHMCP')
    verbose: bool           = field(default=False)
    save_iter_data: bool    = field(default=False)
    time_limit: float       = field(default=None)

    debug: bool             = field(default=False)
    debug_plot: bool        = field(default=False)
    pause_on_plot: bool     = field(default=False)

if __name__ == "__main__":
    pass
