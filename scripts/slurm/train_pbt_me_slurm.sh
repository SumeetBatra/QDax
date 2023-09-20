#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c20
#SBATCH --output=tmp/pbt_me_run_%j.log

# the following params need to be set manually
# PROJECT_PATH: path to QDax root on your system
# ENV_NAME: which environment to run on (see qdax/environments/__init__.py for a list of all runnable envs)
# then from the project root dir you can run ./scripts/train_pga_me.sh

ENV_NAME="halfcheetah_uni"
GRID_SIZE=10  # number of cells per archive dimension

set -- 1111 2222 3333 4444  # seeds

for item in "$@";
 do echo "Running seed $item";
 RUN_NAME="pbt_me_"$ENV_NAME"_baseline_seed_"$item
 echo $RUN_NAME
 srun python -m scripts.train_pbt_me --env_name="$ENV_NAME" \
                                --seed="$item" \
                                --run_name="$RUN_NAME" \
                                --grid_size=$GRID_SIZE \
                                --num_iterations=10000 \
                                --num_loops=625 \
                                --use_wandb=True
done