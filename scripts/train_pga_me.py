# https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/pgame_example.ipynb#scrollTo=Qj3-0q570bNy

import os
import time
import functools
import argparse
import wandb
import shutil

import jax
import jax.numpy as jnp

from attrdict import AttrDict
from distutils.util import strtobool
from utils.utilities import log, config_wandb, get_checkpoints

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax import environments
from qdax.core.neuroevolution.mdp_utils import scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation

from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--policy_hidden_layer_sizes', type=int, action='append')
    parser.add_argument('--iso_sigma', type=float, default=0.005)
    parser.add_argument('--line_sigma', type=float, default=0.05)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--run_name', type=str, default='pga_me_test_run')
    parser.add_argument('--wandb_group', type=str, default='PGA-ME')
    parser.add_argument('--log_period', type=int, default=10)
    # cvt map elites params
    parser.add_argument('--num_init_cvt_samples', type=int, default=50000)
    parser.add_argument('--num_centroids', type=int, default=1024)
    # pga-me params
    parser.add_argument('--min_bd', type=float, default=0.)
    parser.add_argument('--max_bd', type=float, default=1.0)
    parser.add_argument('--proportion_mutation_ga', type=float, default=0.5)
    # td3 params
    parser.add_argument('--env_batch_size', type=int, default=100)
    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--critic_hidden_layer_size', type=int, action='append')
    parser.add_argument('--critic_learning_rate', type=float, default=3e-4)
    parser.add_argument('--greedy_learning_rate', type=float, default=3e-4)
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--reward_scaling', type=float, default=1.0)
    parser.add_argument('--transitions_batch_size', type=int, default=256)
    parser.add_argument('--soft_tau_update', type=float, default=0.005)
    parser.add_argument('--num_critic_training_steps', type=int, default=300)
    parser.add_argument('--num_pg_training_steps', type=int, default=100)

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def run():
    cfg = parse_args()
    if cfg.use_wandb:
        config_wandb(project='QDPPO', entity='qdrl', group=cfg.wandb_group, run_name=cfg.run_name, cfg=cfg)
    for key, val in cfg.items():
        log.debug(f'{key}: {val}')

    # Init environment
    env = environments.create(cfg.env_name, episode_length=cfg.episode_length, ctrl_cost_weight=0.01)

    # Init a random key
    random_key = jax.random.PRNGKey(cfg.seed)

    # Init policy network
    policy_layer_sizes = cfg.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(
            env_state,
            policy_params,
            random_key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=cfg.env_batch_size)
    fake_batch = jnp.zeros(shape=(cfg.env_batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=cfg.env_batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[cfg.env_name]
    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=cfg.episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[cfg.env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * cfg.episode_length,
    )

    # Define the PG-emitter config
    pga_emitter_config = PGAMEConfig(
        env_batch_size=cfg.env_batch_size,
        batch_size=cfg.transitions_batch_size,
        proportion_mutation_ga=cfg.proportion_mutation_ga,
        critic_hidden_layer_size=cfg.critic_hidden_layer_size,
        critic_learning_rate=cfg.critic_learning_rate,
        greedy_learning_rate=cfg.greedy_learning_rate,
        policy_learning_rate=cfg.policy_learning_rate,
        noise_clip=cfg.noise_clip,
        policy_noise=cfg.policy_noise,
        discount=cfg.discount,
        reward_scaling=cfg.reward_scaling,
        replay_buffer_size=cfg.replay_buffer_size,
        soft_tau_update=cfg.soft_tau_update,
        num_critic_training_steps=cfg.num_critic_training_steps,
        num_pg_training_steps=cfg.num_pg_training_steps
    )

    # Get the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=cfg.iso_sigma, line_sigma=cfg.line_sigma
    )

    pg_emitter = PGAMEEmitter(
        config=pga_emitter_config,
        policy_network=policy_network,
        env=env,
        variation_fn=variation_fn,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=pg_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=cfg.num_init_cvt_samples,
        num_centroids=cfg.num_centroids,
        minval=cfg.min_bd,
        maxval=cfg.max_bd,
        random_key=random_key,
    )

    # compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    log_period = cfg.log_period
    num_loops = int(cfg.num_iterations / log_period)

    experiment_dir = './experiments'
    experiment_dir = os.path.join(experiment_dir, cfg.run_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    logdir = os.path.join(experiment_dir, "logs")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    filename = f'{cfg.run_name}.csv'
    filepath = os.path.join(logdir, filename)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    csv_logger = CSVLogger(
        filepath,
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
    )
    all_metrics = {}

    # main loop
    for i in range(num_loops):
        start_time = time.time()
        log.info(f'Loop {i}, Num Loops: {num_loops}, Progress: {(100.0 * (i / num_loops)):.2f}%')
        # main iterations
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            map_elites.scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # save a checkpoint and delete older ones
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{i:05d}/')
        os.mkdir(checkpoint_path)
        repertoire.save(checkpoint_path)
        if len(os.listdir(checkpoint_dir)) > 3:
            oldest_checkpoint_rel_path = list(sorted(os.listdir(checkpoint_dir)))[0]
            oldest_checkpoint = os.path.join(checkpoint_dir, oldest_checkpoint_rel_path)
            shutil.rmtree(oldest_checkpoint)

        # log metrics
        logged_metrics = {"time": timelapse, "loop": 1 + i, "iteration": 1 + i * log_period}
        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]
            log.debug(f'{key}: {logged_metrics[key]}')
            if cfg.use_wandb:
                wandb.log({
                    'loop': i + 1,
                    'iteration': (i + 1) * log_period,
                    f'{key}': logged_metrics[key]
                })

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        csv_logger.log(logged_metrics)


if __name__ == '__main__':
    run()
