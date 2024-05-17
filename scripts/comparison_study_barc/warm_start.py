from exact_dynamic_game import get_exact_dynamic_game

from DGSQP.solvers.IBR import IBR
from DGSQP.solvers.solver_types import IBRParams

from DGSQP.solvers.PID import PIDLaneFollower
from DGSQP.solvers.solver_types import PIDParams

from globals import dt, TOL

import numpy as np
import copy

def get_warm_start_solver(args):
    game_def = get_exact_dynamic_game(args)

    N, joint_model, agent_costs, agent_constrs, shared_constrs, state_input_lb, state_input_ub = game_def

    ibr_params = IBRParams(solver_name='ibr',
                                dt=dt,
                                N=N,
                                line_search_iters=50,
                                ibr_iters=1,
                                use_ps=False,
                                p_tol=TOL,
                                d_tol=TOL,
                                linear_solver='mumps',
                                verbose=False,
                                code_gen=False,
                                jit=False,
                                opt_flag='O3',
                                solver_dir=None,
                                debug_plot=False,
                                pause_on_plot=False)
    ibr_controller = IBR(joint_model, 
                                agent_costs, 
                                agent_constrs,
                                shared_constrs,
                                {'ub': state_input_ub, 'lb': state_input_lb},
                                ibr_params)
    
    def warm_start_solver(joint_state):
        car1_sim_state, car2_sim_state = joint_state
        car1_state_input_min, car2_state_input_min = state_input_lb
        car1_state_input_max, car2_state_input_max = state_input_ub
        car1_dyn_model, car2_dyn_model = joint_model.dynamics_models

        # Set up PID controllers for warm start
        car1_steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                                    x_ref=car1_sim_state.p.x_tran,
                                    u_max=car1_state_input_max.u.u_steer, 
                                    u_min=car1_state_input_min.u.u_steer, 
                                    du_max=10, 
                                    du_min=-10)
        car1_speed_params = PIDParams(dt=dt, Kp=1.0, 
                                    x_ref=car1_sim_state.v.v_long,
                                    u_max=car1_state_input_max.u.u_a, 
                                    u_min=car1_state_input_min.u.u_a, 
                                    du_max=10, 
                                    du_min=-10)
        car1_pid_controller = PIDLaneFollower(dt, car1_steer_params, car1_speed_params)

        car2_steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                                    x_ref=car2_sim_state.p.x_tran,
                                    u_max=car2_state_input_max.u.u_steer, 
                                    u_min=car2_state_input_min.u.u_steer, 
                                    du_max=4.5, 
                                    du_min=-4.5)
        car2_speed_params = PIDParams(dt=dt, Kp=1.0, 
                                    x_ref=car2_sim_state.v.v_long,
                                    u_max=car2_state_input_max.u.u_a, 
                                    u_min=car2_state_input_min.u.u_a, 
                                    du_max=4.5, 
                                    du_min=-4.5)
        car2_pid_controller = PIDLaneFollower(dt, car2_steer_params, car2_speed_params)

        # Construct initial guess for with PID
        car1_state = [copy.deepcopy(car1_sim_state)]
        for _ in range(N):
            state = copy.deepcopy(car1_state[-1])
            car1_pid_controller.step(state)
            car1_dyn_model.step(state)
            car1_state.append(state)
            
        car2_state = [copy.deepcopy(car2_sim_state)]
        for _ in range(N):
            state = copy.deepcopy(car2_state[-1])
            car2_pid_controller.step(state)
            car2_dyn_model.step(state)
            car2_state.append(state)
        
        car1_q_ws = np.zeros((N+1, car1_dyn_model.n_q))
        car2_q_ws = np.zeros((N+1, car2_dyn_model.n_q))
        car1_s_ws = np.zeros(N+1)
        car2_s_ws = np.zeros(N+1)
        car1_u_ws = np.zeros((N, car1_dyn_model.n_u))
        car2_u_ws = np.zeros((N, car2_dyn_model.n_u))
        for k in range(N+1):
            car1_s_ws[k] = car1_state[k].p.s-1e-6
            car2_s_ws[k] = car2_state[k].p.s-1e-6
            # car1_q_ws[k] = np.array([car1_state[k].x.x, car1_state[k].x.y, car1_state[k].v.v_long, car1_state[k].p.e_psi, car1_state[k].p.s-1e-6, car1_state[k].p.x_tran])
            # car2_q_ws[k] = np.array([car2_state[k].x.x, car2_state[k].x.y, car2_state[k].v.v_long, car2_state[k].p.e_psi, car2_state[k].p.s-1e-6, car2_state[k].p.x_tran])
            if k < N:
                car1_u_ws[k] = np.array([car1_state[k+1].u.u_a, car1_state[k+1].u.u_steer])
                car2_u_ws[k] = np.array([car2_state[k+1].u.u_a, car2_state[k+1].u.u_steer])

        # car1_ds_ws = (car1_q_ws[1:,4] - car1_q_ws[:-1,4])/dt
        # car2_ds_ws = (car2_q_ws[1:,4] - car2_q_ws[:-1,4])/dt
        car1_ds_ws = (car1_s_ws[1:] - car1_s_ws[:-1])/dt
        car2_ds_ws = (car2_s_ws[1:] - car2_s_ws[:-1])/dt

        warm_start_success = False

        ibr_controller.set_warm_start([car1_u_ws, car2_u_ws])
        ibr_controller.step([car1_sim_state, car2_sim_state])
        
        if np.all([s.stats()['success'] for s in ibr_controller.br_solvers]):
            print('IBR warm start success')
            warm_start_success = True
            car1_u_ws = ibr_controller.u_pred[:,:car1_dyn_model.n_u]
            car2_u_ws = ibr_controller.u_pred[:,car1_dyn_model.n_u:]
            car1_ds_ws = (ibr_controller.q_pred[1:,4] - ibr_controller.q_pred[:-1,4])/dt
            car2_ds_ws = (ibr_controller.q_pred[1:,10] - ibr_controller.q_pred[:-1,10])/dt

        car1_pa_u_ws = np.hstack([car1_u_ws, car1_ds_ws.reshape((-1,1))])
        car2_pa_u_ws = np.hstack([car2_u_ws, car2_ds_ws.reshape((-1,1))])

        exact_ws = [car1_u_ws, car2_u_ws]
        approximate_ws = [car1_pa_u_ws, car2_pa_u_ws]

        if args.game_type == 'exact':
            return exact_ws
        elif args.game_type == 'approximate':
            return approximate_ws
    
    return warm_start_solver