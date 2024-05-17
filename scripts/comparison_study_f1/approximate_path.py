from DGSQP.solvers.PATHMCP_frenet_approx import PATHMCP

from globals import PATH_PARAMS

def get_approximate_path_solver(args, game_def):
    N, joint_model, agent_costs, agent_constrs, shared_constrs, state_input_lb, state_input_ub = game_def

    PATH_PARAMS.N = N
    PATH_PARAMS.nms = not args.no_nms

    solver = PATHMCP(joint_model, 
                        agent_costs, 
                        agent_constrs,
                        shared_constrs,
                        {'ub': state_input_ub, 'lb': state_input_lb},
                        PATH_PARAMS)
    
    return solver