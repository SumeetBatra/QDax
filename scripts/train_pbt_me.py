import argparse
import functools
import time
import os
import shutil
import wandb
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.io import html
from tqdm import tqdm
from attrdict import AttrDict
from distutils.util import strtobool
from utils.utilities import log, config_wandb, get_checkpoints

from qdax import environments
from qdax.baselines.sac_pbt import PBTSAC, PBTSacConfig, PBTSacTrainingState
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.distributed_map_elites import DistributedMAPElites
from qdax.core.emitters.pbt_me_emitter import PBTEmitter, PBTEmitterConfig
from qdax.core.emitters.pbt_variation_operators import sac_pbt_variation_fn
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--run_name', type=str, default='pbt_me_test_run')
    parser.add_argument('--wandb_group', type=str, default='PBT-ME')
    parser.add_argument('--env_name', type=str)
    # SAC config
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha_init', type=float, default=1.0)
    parser.add_argument('--fix_alpha', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--normalize_observations', type=lambda x: bool(strtobool(x)), default=False)

    # emitter config
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--pg_population_size_per_device', type=int, default=10)
    parser.add_argument('--ga_population_size_per_device', type=int, default=30)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--env_batch_size', type=int, default=250)
    parser.add_argument('--grad_updates_per_step', type=float, default=1.0)
    parser.add_argument('--iso_sigma', type=float, default=0.005)
    parser.add_argument('--line_sigma', type=float, default=0.05)

    # pbt params
    parser.add_argument('--fraction_best_to_replace_from', type=float, default=0.1)
    parser.add_argument('--fraction_to_replace_from_best', type=float, default=0.2)
    parser.add_argument('--fraction_to_replace_from_samples', type=float, default=0.4)

    parser.add_argument('--eval_env_batch_size', default=1)

    # Map elites config
    parser.add_argument('--grid_size', type=int)
    parser.add_argument('--log_period', type=int, default=1)
    parser.add_argument('--num_loops', type=int, default=10)

    parser.add_argument('--load_repertoire_from_cp', type=str, default=None)
    parser.add_argument('--iter', type=int, default=0)

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def load_from_checkpoint(cp_path, policy_network, obs_size, random_key):
    # Init population of policies
    random_key, subkey = jax.random.split(random_key)
    fake_batch = jnp.zeros(shape=(obs_size,))
    fake_params = policy_network.init(subkey, fake_batch)

    _, reconstruction_fn = ravel_pytree(fake_params)

    repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=cp_path)
    return repertoire


def run():
    devices = jax.devices("gpu")
    num_devices = len(devices)
    print(f"Detected the following {num_devices} device(s): {devices}")

    cfg = parse_args()
    if cfg.use_wandb:
        config_wandb(project='QDPPO', entity='qdrl', group=cfg.wandb_group, run_name=cfg.run_name, cfg=cfg)
    for key, val in cfg.items():
        log.debug(f'{key}: {val}')

    hidden_layer_sizes = (128, 128)

    env_step_multiplier = (
                                  (cfg.pg_population_size_per_device + cfg.ga_population_size_per_device)
                                  * cfg.eval_env_batch_size
                                  * cfg.episode_length
                                  + cfg.num_iterations * cfg.pg_population_size_per_device
                          ) * num_devices

    env = environments.create(
        env_name=cfg.env_name,
        batch_size=cfg.env_batch_size * cfg.pg_population_size_per_device,
        episode_length=cfg.episode_length,
        auto_reset=True,
    )

    eval_env = environments.create(
        env_name=cfg.env_name,
        batch_size=cfg.eval_env_batch_size,
        episode_length=cfg.episode_length,
        auto_reset=True,
    )

    min_bd, max_bd = env.behavior_descriptor_limits

    key = jax.random.PRNGKey(cfg.seed)
    key, subkey = jax.random.split(key)
    env_states = jax.jit(env.reset)(rng=subkey)
    eval_env_first_states = jax.jit(eval_env.reset)(rng=subkey)

    # get agent
    config = PBTSacConfig(
        batch_size=cfg.batch_size,
        episode_length=cfg.episode_length,
        tau=cfg.tau,
        normalize_observations=cfg.normalize_observations,
        alpha_init=cfg.alpha_init,
        hidden_layer_sizes=hidden_layer_sizes,
        fix_alpha=cfg.fix_alpha,
    )

    agent = PBTSAC(config=config, action_size=env.action_size)

    # init emitter
    emitter_config = PBTEmitterConfig(
        buffer_size=cfg.buffer_size,
        num_training_iterations=cfg.num_iterations // cfg.env_batch_size,
        env_batch_size=cfg.env_batch_size,
        grad_updates_per_step=cfg.grad_updates_per_step,
        pg_population_size_per_device=cfg.pg_population_size_per_device,
        ga_population_size_per_device=cfg.ga_population_size_per_device,
        num_devices=num_devices,
        fraction_best_to_replace_from=cfg.fraction_best_to_replace_from,
        fraction_to_replace_from_best=cfg.fraction_to_replace_from_best,
        fraction_to_replace_from_samples=cfg.fraction_to_replace_from_samples,
        fraction_sort_exchange=0.1,
    )

    variation_fn = functools.partial(
        sac_pbt_variation_fn, iso_sigma=cfg.iso_sigma, line_sigma=cfg.line_sigma
    )

    emitter = PBTEmitter(
        pbt_agent=agent,
        config=emitter_config,
        env=env,
        variation_fn=variation_fn,
    )

    # get scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[cfg.env_name]
    eval_policy = agent.get_eval_qd_fn(eval_env, bd_extraction_fn=bd_extraction_fn)

    def scoring_function(genotypes, random_key):
        population_size = jax.tree_util.tree_leaves(genotypes)[0].shape[0]
        first_states = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), eval_env_first_states
        )
        first_states = jax.tree_map(
            lambda x: jnp.repeat(x, population_size, axis=0), first_states
        )
        population_returns, population_bds, _, _ = eval_policy(genotypes, first_states)
        return population_returns, population_bds, None, random_key

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[cfg.env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * cfg.episode_length,
    )

    # Get the MAP-Elites algorithm
    map_elites = DistributedMAPElites(
        scoring_function=scoring_function,
        emitter=emitter,
        metrics_function=metrics_function,
    )

    grid_shape = (cfg.grid_size,) * env.behavior_descriptor_length
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_bd,
        maxval=max_bd,
    )

    key, *keys = jax.random.split(key, num=1 + num_devices)
    keys = jnp.stack(keys)

    # get the initial training states and replay buffers
    agent_init_fn = agent.get_init_fn(
        population_size=cfg.pg_population_size_per_device + cfg.ga_population_size_per_device,
        action_size=env.action_size,
        observation_size=env.observation_size,
        buffer_size=cfg.buffer_size,
    )
    keys, training_states, _ = jax.pmap(agent_init_fn, axis_name="p", devices=devices)(keys)

    # empty optimizers states to avoid too heavy repertories
    training_states = jax.pmap(
        jax.vmap(training_states.__class__.empty_optimizers_states),
        axis_name="p",
        devices=devices,
    )(training_states)

    # initialize map-elites
    repertoire, emitter_state, keys = map_elites.get_distributed_init_fn(
        devices=devices, centroids=centroids
    )(init_genotypes=training_states, random_key=keys)

    update_fn = map_elites.get_distributed_update_fn(
        num_iterations=cfg.log_period, devices=devices
    )

    experiment_dir = './experiments'
    experiment_dir = os.path.join(experiment_dir, cfg.run_name)
    if cfg.load_repertoire_from_cp is None:
        assert not os.path.exists(experiment_dir), f'Error: {experiment_dir=} already exists. Danger of overriding ' \
                                                   f'existing experiment.'
    os.makedirs(experiment_dir, exist_ok=True)

    logdir = os.path.join(experiment_dir, "logs")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    filename = f'{cfg.run_name}.csv'
    filepath = os.path.join(logdir, filename)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    all_metrics = {}
    starting_iter = cfg.iter // cfg.log_period

    num_loops = cfg.num_loops
    for i in tqdm(range(num_loops // cfg.log_period), total=num_loops // cfg.log_period):
        start_time = time.time()

        repertoire, emitter_state, keys, metrics = update_fn(
            repertoire, emitter_state, keys
        )
        metrics_cpu = jax.tree_map(
            lambda x: jax.device_put(x, jax.devices("cpu")[0])[0], metrics
        )
        timelapse = time.time() - start_time

        # save a checkpoint and delete older ones
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{i:05d}/')
        os.mkdir(checkpoint_path)
        repertoire.save(checkpoint_path)
        if len(os.listdir(checkpoint_dir)) > 2:
            oldest_checkpoint_rel_path = list(sorted(os.listdir(checkpoint_dir)))[0]
            oldest_checkpoint = os.path.join(checkpoint_dir, oldest_checkpoint_rel_path)
            shutil.rmtree(oldest_checkpoint)

        # log metrics
        logged_metrics = {"time": timelapse, "loop": 1 + i, "iteration": 1 + i * cfg.log_period}
        for key, value in metrics_cpu.items():
            logged_metrics[key] = value[-1]
            log.debug(f'{key}: {logged_metrics[key]}')
            if cfg.use_wandb:
                wandb.log({
                    'loop': i + 1,
                    'iteration': (i + 1) * cfg.log_period,
                    f'{key}': logged_metrics[key]
                })
            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value


if __name__ == '__main__':
    run()
