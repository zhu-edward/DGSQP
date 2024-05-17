OUT_DIR=~/results/regularization_study_barc_kinematic_$(date +%Y-%m-%d_%H-%M-%S)

M=100
N=25

for reg in 1 10 100 1000
do
    for decay in 0.5 0.65 0.8 0.95 1
    do
        # Approximate DGSQP 1 SQP evalution
        python3 monte_carlo_main.py \
            --samples $M \
            --N $N \
            --solver dgsqp \
            --dynamics_type kinematic \
            --game_type approximate \
            --merit_function stat_l1 \
            --merit_decrease_condition armijo \
            --merit_parameter adaptive \
            --reg_init $reg \
            --reg_decay $decay \
            --eval_type once \
            --out_dir $OUT_DIR

        # Approximate DGSQP multiple SQP evalution
        python3 monte_carlo_main.py \
            --samples $M \
            --N $N \
            --solver dgsqp \
            --dynamics_type kinematic \
            --game_type approximate \
            --merit_function stat_l1 \
            --merit_decrease_condition armijo \
            --merit_parameter adaptive \
            --reg_init $reg \
            --reg_decay $decay \
            --eval_type always \
            --out_dir $OUT_DIR
        done
done

# Approximate DGSQP 1 SQP evalution
python3 monte_carlo_main.py \
    --samples $M \
    --N $N \
    --solver dgsqp \
    --dynamics_type kinematic \
    --game_type approximate \
    --merit_function stat_l1 \
    --merit_decrease_condition armijo \
    --merit_parameter adaptive \
    --reg_init 0 \
    --reg_decay 1 \
    --eval_type once \
    --out_dir $OUT_DIR

# Approximate DGSQP multiple SQP evalution
python3 monte_carlo_main.py \
    --samples $M \
    --N $N \
    --solver dgsqp \
    --dynamics_type kinematic \
    --game_type approximate \
    --merit_function stat_l1 \
    --merit_decrease_condition armijo \
    --merit_parameter adaptive \
    --reg_init 0 \
    --reg_decay 1 \
    --eval_type always \
    --out_dir $OUT_DIR