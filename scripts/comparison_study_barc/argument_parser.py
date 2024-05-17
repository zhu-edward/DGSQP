import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', dest='N', type=int)
parser.add_argument('--solver', dest='solver')
parser.add_argument('--dynamics_type', dest='dynamics_type')
parser.add_argument('--game_type', dest='game_type')
parser.add_argument('--eval_type', dest='eval_type')
parser.add_argument('--merit_function', dest='merit_function')
parser.add_argument('--merit_decrease_condition', dest='merit_decrease_condition')
parser.add_argument('--merit_parameter', dest='merit_parameter')
parser.add_argument('--reg_init', dest='reg_init', type=float)
parser.add_argument('--reg_decay', dest='reg_decay', type=float)
parser.add_argument('--no_nms', action='store_true', default=False)
parser.add_argument('--cost_setting', dest='cost_setting', type=int, default=0)
parser.add_argument('--samples', dest='samples', type=int)
parser.add_argument('--out_dir', dest='out_dir')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--start_idx', dest='start_idx', type=int)