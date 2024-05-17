OUT_DIR=~/results/ablation_study_barc_kinematic_$(date +%Y-%m-%d_%H-%M-%S)

M=10

for N in 15
do
    # Exact DGSQP
    # python3 monte_carlo_main.py \
    #     --samples $M \
    #     --N $N \
    #     --solver dgsqp \
    #     --dynamics_type kinematic \
    #     --game_type exact \
    #     --merit_function stat_l1 \
    #     --merit_decrease_condition armijo \
    #     --merit_parameter adaptive \
    #     --cost_setting 1 \
    #     --out_dir $OUT_DIR

    python3 monte_carlo_main.py \
        --samples $M \
        --N $N \
        --solver dgsqp \
        --dynamics_type kinematic \
        --game_type exact \
        --merit_function sum_obj_l1 \
        --merit_decrease_condition armijo \
        --merit_parameter adaptive \
        --cost_setting 1 \
        --out_dir $OUT_DIR

    # python3 monte_carlo_main.py \
    #     --samples $M \
    #     --N $N \
    #     --solver dgsqp \
    #     --dynamics_type kinematic \
    #     --game_type exact \
    #     --merit_function stat_l1 \
    #     --merit_decrease_condition armijo \
    #     --merit_parameter adaptive \
    #     --no_nms \
    #     --cost_setting 1 \
    #     --out_dir $OUT_DIR

    # python3 monte_carlo_main.py \
    #     --samples $M \
    #     --N $N \
    #     --solver dgsqp \
    #     --dynamics_type kinematic \
    #     --game_type exact \
    #     --merit_function sum_obj_l1 \
    #     --merit_decrease_condition armijo \
    #     --merit_parameter adaptive \
    #     --no_nms \
    #     --cost_setting 1 \
    #     --out_dir $OUT_DIR

    # python3 monte_carlo_main.py \
    #     --samples $M \
    #     --N $N \
    #     --solver path \
    #     --dynamics_type kinematic \
    #     --game_type exact \
    #     --cost_setting 1 \
    #     --out_dir $OUT_DIR

done