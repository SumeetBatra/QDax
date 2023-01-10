#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c20
#SBATCH --output=tmp/pga_me_halfcheetah_uni_%j.log

ENV_NAME="halfcheetah_uni"
GRID_SIZE=50

set -- 1111 2222 3333 4444  # seeds

for item in "$@";
 do echo "Running seed $item";
 RUN_NAME="pga_me_"$ENV_NAME"_baseline_seed_"$item"_v2"
 echo $RUN_NAME
 srun python -m scripts.train_pga_me --env_name="$ENV_NAME" \
                                 --seed="$item" \
                                 --run_name=$RUN_NAME \
                                 --grid_size=$GRID_SIZE \
                                 --episode_length=1000 \
                                 --num_iterations=4000 \
                                 --policy_hidden_layer_sizes=128 \
                                 --policy_hidden_layer_sizes=128 \
                                 --use_wandb=True \
                                 --critic_hidden_layer_size=256 \
                                 --critic_hidden_layer_size=256 \
                                 --env_batch_size=300
done