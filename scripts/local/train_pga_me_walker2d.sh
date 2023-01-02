#!/bin/sh

# the following params need to be set manually
# PROJECT_PATH: path to QDax root on your system
# ENV_NAME: which environment to run on (see qdax/environments/__init__.py for a list of all runnable envs)
# then from the project root dir you can run ./scripts/train_pga_me.sh

PROJECT_PATH="/home/sumeet/QDax/"
ENV_NAME="walker2d_uni"
NUM_CENTROIDS=2500

BASE_CLI_ARGS="--episode_length=1000 \
              --num_iterations=4000 \
              --policy_hidden_layer_sizes=128 \
              --policy_hidden_layer_sizes=128 \
              --use_wandb=True \
              --critic_hidden_layer_size=256 \
              --critic_hidden_layer_size=256 \
              --env_batch_size=300"

set -- 1111 2222 3333 4444  # seeds

for item in "$@";
 do echo "Running seed $item";
 RUN_NAME="pga_me_"$ENV_NAME"_baseline_seed"$item
 echo $RUN_NAME
 python -m scripts.train_pga_me --env_name="$ENV_NAME" \
                                 --seed="$item" \
                                 --run_name=$RUN_NAME \
                                 --num_centroids=$NUM_CENTROIDS
                                  $BASE_CLI_ARGS
done

#python -m scripts.train_pga_me BASE_CLI_ARGS