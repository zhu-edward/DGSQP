OUT_DIR=~/results/comparison_study_f1_dynamic_$(date +%Y-%m-%d_%H-%M-%S)

M=100

for N in 15 20 25
do
    # Approximate DGSQP
    python3 monte_carlo_main.py \
        --samples $M \
        --N $N \
        --solver dgsqp \
        --dynamics_type dynamic \
        --game_type approximate \
        --merit_function stat_l1 \
        --merit_decrease_condition armijo \
        --merit_parameter adaptive \
        --eval_type once \
        --out_dir $OUT_DIR

    python3 monte_carlo_main.py \
        --samples $M \
        --N $N \
        --solver dgsqp \
        --dynamics_type dynamic \
        --game_type approximate \
        --merit_function stat_l1 \
        --merit_decrease_condition armijo \
        --merit_parameter adaptive \
        --eval_type always \
        --out_dir $OUT_DIR

    # Approximate PATH
    python3 monte_carlo_main.py \
        --samples $M \
        --N $N \
        --solver path \
        --dynamics_type dynamic \
        --game_type approximate \
        --out_dir $OUT_DIR
done