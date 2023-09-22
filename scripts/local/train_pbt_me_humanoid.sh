ENV_NAME="humanoid_uni"
GRID_SIZE=10  # number of cells per archive dimension

set -- 1111 2222 3333 4444  # seeds

for item in "$@";
 do echo "Running seed $item";
 RUN_NAME="pbt_me_"$ENV_NAME"_baseline_seed_"$item
 echo $RUN_NAME
 python -m scripts.train_pbt_me --env_name="$ENV_NAME" \
                                --seed="$item" \
                                --run_name="$RUN_NAME" \
                                --grid_size=$GRID_SIZE \
                                --num_iterations=10000 \
                                --num_loops=625 \
                                --use_wandb=True
done