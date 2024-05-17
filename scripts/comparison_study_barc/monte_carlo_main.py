from argument_parser import parser

import numpy as np
import pathlib
import pickle
import pdb

def main():
    args = parser.parse_args()

    save_results = args.save
    start_idx = args.start_idx

    print(f'N = {args.N}')
    print(f'Solving {args.game_type} dynamic game with {args.solver}')
    print(f'Dynamics type: {args.dynamics_type}')
    print(f'Eval type: {args.eval_type}')
    print(f'Merit function: {args.merit_function}')
    print(f'Merit decrease condition: {args.merit_decrease_condition}')
    print(f'Merit parameter: {args.merit_parameter}')
    print(f'Regularization init: {args.reg_init}')
    print(f'Regularization decay: {args.reg_decay}')
    print(f'Nonmonotone line search: {not args.no_nms}')
    if start_idx is not None:
        print(f'Starting from sample {start_idx}')
    if save_results:
        print(f'Saving to {args.out_dir}')
    print('==============================================================')

    _dir = f'N{args.N}-{args.dynamics_type}-{args.game_type}-{args.solver}'
    if args.eval_type:
        _dir += f'-{args.eval_type}'
    if args.merit_function:
        _dir += f'-{args.merit_function}'
    if args.merit_decrease_condition:
        _dir += f'-{args.merit_decrease_condition}'
    if args.merit_parameter:
        _dir += f'-{args.merit_parameter}'
    if args.reg_init is not None:
        _dir += f'-reg_{args.reg_init}'
    if args.reg_decay is not None:
        _dir += f'-decay_{args.reg_decay}'
    if args.no_nms:
        _dir += '-no_nms'
    else:
        _dir += '-nms'
    
    if save_results:
        out_dir = pathlib.Path(args.out_dir, _dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

    N = int(args.N)
    M = int(args.samples)
    
    if args.game_type == 'exact':
        if args.dynamics_type == 'kinematic':
            from exact_dynamic_game import get_exact_dynamic_game
        elif args.dynamics_type == 'dynamic':
            from exact_dynamic_game_dynamic import get_exact_dynamic_game
        game_def = get_exact_dynamic_game(args)
        if args.solver == 'dgsqp':
            from exact_dgsqp import get_exact_dgsqp_solver
            solver = get_exact_dgsqp_solver(args, game_def)
        elif args.solver == 'path':
            from exact_path import get_exact_path_solver
            solver = get_exact_path_solver(args, game_def)
    elif args.game_type == 'approximate':
        if args.dynamics_type == 'kinematic':
            from approximate_dynamic_game import get_approximate_dynamic_game
        elif args.dynamics_type == 'dynamic':
            from approximate_dynamic_game_dynamic import get_approximate_dynamic_game
        game_def = get_approximate_dynamic_game(args)
        if args.solver == 'dgsqp':
            from approximate_dgsqp import get_approximiate_dgsqp_solver
            solver = get_approximiate_dgsqp_solver(args, game_def)
        elif args.solver == 'path':
            from approximate_path import get_approximate_path_solver
            solver = get_approximate_path_solver(args, game_def)

    if args.dynamics_type == 'kinematic':
        from warm_start import get_warm_start_solver
        from monte_carlo_sampler import get_sample
    elif args.dynamics_type == 'dynamic':
        from warm_start_dynamic import get_warm_start_solver
        from monte_carlo_sampler_dynamic import get_sample
    
    warm_start_solver = get_warm_start_solver(args)

    i = 0
    
    while i < M:
        print(f'Sample {i+1}/{M}')
        joint_state = get_sample()

        ws = warm_start_solver(joint_state)

        if ws is None:
            print('Warm start failed, resampling...')
            continue

        solver.set_warm_start(np.hstack(ws))

        if start_idx is not None:
            if i < start_idx:
                i += 1
                continue

        info = solver.solve(joint_state)
        result = dict(solver_result=info,
                      initial_condition=joint_state)
        
        if save_results:
            file_name = f'sample_{i+1}.pkl'
            out_path = out_dir.joinpath(file_name)
            with open(out_path, 'wb') as f:
                pickle.dump(result, f)

        i += 1

        print('==============================================================')

if __name__ == '__main__':
    main()