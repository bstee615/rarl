import logging

import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from arguments import parse_args
from main import run


def get_mean_reward_last_n_steps(n, log_dir):
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(x) > 0:
        # Mean training reward over the last 100 episodes
        mean_reward = np.mean(y[-n:])
        return mean_reward
    else:
        logging.warning(f'{get_mean_reward_last_n_steps.__name__} called when the number of logged timesteps was 0')


def add_config_args(args, config):
    """
    Returns a Namespace object with this config's hyperparameters mixed in.
    """
    return args


def monitor_dir(envname, config):
    """
    Returns a directory named for this config's hyperparams
    """
    return f'tmp-{envname}-{config["adv_force"]}'


def trainable(config):
    args = add_config_args(config["args"], config)
    args.monitor_dir = monitor_dir(config["envname"], config)

    def evaluate(_):
        # TODO may have to do something to prevent too-early stopping
        tune.report(reward=get_mean_reward_last_n_steps(config["evaluate_mean_n"], args.monitor_dir))

    run(args, evaluate_fn=evaluate)


def main():
    num_samples = 10
    evaluate_mean_n = 1000  # Number of timesteps over which to evaluate the mean reward

    logging.basicConfig(level=logging.INFO)
    envname = 'AdversarialAntBulletEnv-v0'
    trainingconfig_name = 'original-million'
    args = parse_args(['--name', trainingconfig_name, '--env', envname])
    config = {
        "adv_force": tune.uniform(0, 1),  # TODO set range
        "args": args,
        "envname": envname,
        "evaluate_mean_n": evaluate_mean_n,
    }

    # TODO set search and sched
    search = HyperOptSearch()
    sched = ASHAScheduler()

    # Pass in a Trainable class or function to tune.run.
    anal = tune.run(trainable,
                    config=config,
                    num_samples=num_samples,
                    scheduler=sched,
                    search_alg=search,
                    metric="reward",
                    mode="max")


if __name__ == '__main__':
    main()
