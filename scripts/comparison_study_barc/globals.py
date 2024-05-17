from DGSQP.solvers.solver_types import DGSQPV2Params as DGSQPParams
from DGSQP.solvers.solver_types import PATHMCPParams


TRACK = 'L_track_barc'

TOL = 1e-4

dt = 0.1

CAR1_R = 0.23
CAR2_R = 0.23

VL = 0.37
VW = 0.195

DISCRETIZATION_METHOD = 'rk4'
M = 10

PATH_PARAMS = PATHMCPParams(solver_name='PATH',
                                dt=dt,
                                p_tol=TOL,
                                verbose=False)

DGSQP_PARAMS = DGSQPParams(solver_name='DGSQP',
                        dt=dt,
                        nms=True,
                        nms_frequency=10,
                        nms_memory_size=10,
                        line_search_iters=20,
                        sqp_iters=500,
                        p_tol=TOL,
                        d_tol=TOL,
                        reg=1e2,
                        reg_decay=0.95,
                        delta_decay=0.99,
                        merit_decrease=0.01,
                        beta=0.01,
                        tau=0.5,
                        time_limit=600,
                        verbose=False,
                        code_gen=False,
                        jit=False,
                        opt_flag='O3',
                        solver_dir=None, #'/home/edward-zhu/.mpclab_controllers/DGSQP',
                        so_name='DGSQP.so',
                        qp_interface='casadi',
                        # qp_solver='cplex',
                        qp_solver='osqp',
                        hessian_approximation='none',
                        debug_plot=False,
                        pause_on_plot=False)