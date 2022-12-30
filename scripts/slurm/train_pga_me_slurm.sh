#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c20
#SBATCH --output=tmp/pga_me_ant_uni_%j.log

ENV_NAME="ant_uni"
NUM_CENTROIDS=10000

srun python -m scripts.train_pga_me --env_name="$ENV_NAME" \
                                    --seed=0000 \
                                    --run_name=pga_me_ant_uni \
                                    --num_centroids=$NUM_CENTROIDS \
                                    --episode_length=1000 \
                                    --num_iterations=10000 \
                                    --policy_hidden_layer_sizes=128 \
                                    --policy_hidden_layer_sizes=128 \
                                    --use_wandb=True \
                                    --critic_hidden_layer_size=256 \
                                    --critic_hidden_layer_size=256 \
                                    --env_batch_size=100 \
                                    --ctrl_cost_weight=0.001
