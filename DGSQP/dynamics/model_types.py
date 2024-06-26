#!/usr/bin python3

from dataclasses import dataclass, field
import numpy as np

from DGSQP.types import PythonMsg

@dataclass
class ModelConfig(PythonMsg):
    model_name: str                 = field(default = 'model')
    use_mx: bool                    = field(default = False)
    enable_jacobians: bool          = field(default = True)
    compute_hessians: bool          = field(default = False)
    verbose: bool                   = field(default = False)
    code_gen: bool                  = field(default = False)
    jit: bool                       = field(default = True)
    opt_flag: str                   = field(default = 'O0')
    install: bool                   = field(default = True)
    install_dir: str                = field(default = '~/.dgsqp_models')

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
    wheel_dist_front: float         = field(default = 0.13)
    wheel_dist_rear: float          = field(default = 0.13)
    wheel_dist_center_front: float  = field(default = 0.1)
    wheel_dist_center_rear:  float  = field(default = 0.1)
    bump_dist_front: float          = field(default = 0.15)
    bump_dist_rear: float           = field(default = 0.15)
    bump_dist_center: float         = field(default = 0.1)
    bump_dist_top: float            = field(default = 0.1)
    com_height: float               = field(default = 0.05)

    mass: float                     = field(default = 2.2187)
    gravity: float                  = field(default = 9.81)

    yaw_inertia: float              = field(default = 0.02723)
    pitch_inertia: float            = field(default = 0.03)  # Not being used in dynamics
    roll_inertia: float             = field(default = 0.03)  # Not being used in dynamics
    
    drag_coefficient: float         = field(default = 0.0)  # .05
    damping_coefficient: float      = field(default = 0.0)
    rolling_resistance: float       = field(default = 0.0)
    rolling_resistance_exponent: float = field(default = 0.0)

    tire_model: str                 = field(default = 'pacejka')
    drive_wheels: str               = field(default = 'all')

    wheel_friction: float           = field(default = 0.9)
    pacejka_b_front: float          = field(default = 5.0)
    pacejka_b_rear: float           = field(default = 5.0)
    pacejka_c_front: float          = field(default = 2.28)
    pacejka_c_rear: float           = field(default = 2.28)
    pacejka_d_front: float          = field(default = None)
    pacejka_d_rear: float           = field(default = None)

    linear_bf: float                = field(default = 1.0)
    linear_br: float                = field(default = 1.0)

    simple_slip: bool               = field(default=False)

    def __post_init__(self):
        if self.pacejka_d_front is None:
            self.pacejka_d_front = self.wheel_friction*self.mass*self.gravity * self.wheel_dist_rear / (self.wheel_dist_rear + self.wheel_dist_front)
        if self.pacejka_d_rear is None:
            self.pacejka_d_rear  = self.wheel_friction*self.mass*self.gravity * self.wheel_dist_front / (self.wheel_dist_rear + self.wheel_dist_front)

    def plot_lateral_tire_model(self):
        import matplotlib.pyplot as plt
        alpha = np.linspace(-15, 15, 200)*np.pi/180
        fy = self.pacejka_d_front * np.sin(self.pacejka_c_front * np.arctan(self.pacejka_b_front * alpha))

        plt.plot(alpha, fy)
        plt.show()
        
@dataclass
class KinematicBicycleConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    wheel_dist_front: float         = field(default = 0.13)
    wheel_dist_rear: float          = field(default = 0.13)
    wheel_dist_center_front: float  = field(default = 0.1)
    wheel_dist_center_rear:  float  = field(default = 0.1)
    bump_dist_front: float          = field(default = 0.15)
    bump_dist_rear: float           = field(default = 0.15)
    bump_dist_center: float         = field(default = 0.1)
    bump_dist_top: float            = field(default = 0.1)
    com_height: float               = field(default = 0.05)

    mass: float                     = field(default = 2.366)

    drag_coefficient: float         = field(default = 0.0)
    damping_coefficient: float      = field(default = 0.0)
    slip_coefficient: float         = field(default = 0.0)
    rolling_resistance: float       = field(default = 0.0)
    rolling_resistance_exponent: float = field(default = 0.5)

@dataclass
class PointMassConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    mass: float                         = field(default = 2.366)
    damping_coefficient: float          = field(default = 0.0)  # .05
    drag_coefficient: float             = field(default = 0.0)  # .05
    rolling_resistance: float           = field(default = 0.0)
    rolling_resistance_exponent: float  = field(default = 0.5)

@dataclass
class UnicycleConfig(DynamicsConfig):  # configurations for simulated vehicle model, can grow to be used elsewhere.
    mass: float                         = field(default = 2.366)
    damping_coefficient: float          = field(default = 0.0)  # .05
    drag_coefficient: float             = field(default = 0.0)  # .05
    rolling_resistance: float           = field(default = 0.0)
    rolling_resistance_exponent: float  = field(default = 0.5)

@dataclass
class MultiAgentModelConfig(DynamicsConfig):
    use_mx: bool                    = field(default = False)
