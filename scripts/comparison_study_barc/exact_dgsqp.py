from DGSQP.solvers.DGSQP_v2 import DGSQP

from globals import DGSQP_PARAMS

def get_exact_dgsqp_solver(args, game_def):
    N, joint_model, agent_costs, agent_constrs, shared_constrs, state_input_lb, state_input_ub = game_def

    if args.merit_parameter == 'adaptive':
        merit_parameter = None
    elif args.merit_parameter == 'constant':
        merit_parameter = 1.0

    if args.dynamics_type == 'dynamic':
        use_mx = True
    else:
        use_mx = False

    DGSQP_PARAMS.N = N
    DGSQP_PARAMS.merit_parameter = merit_parameter
    DGSQP_PARAMS.merit_function = args.merit_function
    DGSQP_PARAMS.merit_decrease_condition = args.merit_decrease_condition
    DGSQP_PARAMS.approximation_eval = args.eval_type
    DGSQP_PARAMS.nms = not args.no_nms

    if args.reg_init is not None:
        DGSQP_PARAMS.reg = float(args.reg_init)
    if args.reg_decay is not None:
        DGSQP_PARAMS.reg_decay = float(args.reg_decay)
        
    solver = DGSQP(joint_model,
                    agent_costs, 
                    agent_constrs,
                    shared_constrs,
                    {'ub': state_input_ub, 'lb': state_input_lb},
                    DGSQP_PARAMS,
                    use_mx=use_mx)
    
    return solver