#!/usr/bin python3

from dataclasses import dataclass, field
import numpy as np

from DGSQP.types import PythonMsg

@dataclass
class ModelConfig(PythonMsg):
    model_name: str                 = field(default = 'model')

    enable_jacobians: bool          = field(default = True)
    compute_hessians: bool          = field(default = False)
    verbose: bool                   = field(default = False)
    code_gen: bool                  = field(default = False)
    jit: bool                       = field(default = True)
    opt_flag: str                   = field(default = 'O0')
    install: bool                   = field(default = True)
    install_dir: str                = field(default = '~/.mpclab_common/models')

@dataclass
class DynamicsConfig(ModelConfig):
    track_name: str                 = field(default = None)

    dt: float                       = field(default = 0.01)   # interval of an entire simulation step
    discretization_method: str      = field(default = 'euler')
    M: int                          = field(default = 10) # RK4 integration steps

    # Flag indicating whether dynamics are affected by exogenous noise
    noise: bool                     = field(default = False)
    noise_cov: np.ndarray           = field(default = None)

@dataclass
class DynamicBicycleConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    wheel_dist_front: float         = field(default = 0.125)
    wheel_dist_rear: float          = field(default = 0.125)
    wheel_dist_center_front: float  = field(default = 0.1)
    wheel_dist_center_rear:  float  = field(default = 0.1)
    bump_dist_front: float          = field(default = 0.15)
    bump_dist_rear: float           = field(default = 0.15)
    bump_dist_center: float         = field(default = 0.1)
    bump_dist_top: float            = field(default = 0.1)
    com_height: float               = field(default = 0.05)
    mass: float                     = field(default = 1.98)
    yaw_inertia: float              = field(default = 0.03)
    pitch_inertia: float            = field(default = 0.03)
    roll_inertia: float             = field(default = 0.03)
    gravity: float                  = field(default = 9.81)
    wheel_friction: float           = field(default = 0.8)
    drag_coefficient: float         = field(default = 0.05)
    slip_coefficient: float         = field(default = 0.1)
    pacejka_b: float                = field(default = 1.0)
    pacejka_c: float                = field(default = 1.25)
    pacejka_d_front: float          = field(default = None)
    pacejka_d_rear: float           = field(default = None)

    def __post_init__(self):
        if self.pacejka_d_front is None:
            self.pacejka_d_front = self.wheel_friction*self.mass*self.gravity * self.wheel_dist_rear / (self.wheel_dist_rear + self.wheel_dist_front)
        if self.pacejka_d_rear is None:
            self.pacejka_d_rear  = self.wheel_friction*self.mass*self.gravity * self.wheel_dist_front / (self.wheel_dist_rear + self.wheel_dist_front)

@dataclass
class KinematicBicycleConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    wheel_dist_front: float         = field(default = 0.125)
    wheel_dist_rear: float          = field(default = 0.125)
    wheel_dist_center_front: float  = field(default = 0.1)
    wheel_dist_center_rear:  float  = field(default = 0.1)
    bump_dist_front: float          = field(default = 0.15)
    bump_dist_rear: float           = field(default = 0.15)
    bump_dist_center: float         = field(default = 0.1)
    bump_dist_top: float            = field(default = 0.1)
    com_height: float               = field(default = 0.05)
    mass: float                     = field(default = 1.98)
    gravity: float                  = field(default = 9.81)
    drag_coefficient: float         = field(default = 0.05)
    slip_coefficient: float         = field(default = 0.1)

@dataclass
class MultiAgentModelConfig(DynamicsConfig):
    use_mx: bool                    = field(default = False)
