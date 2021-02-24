import logging
from pathlib import Path

import numpy as np
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from arguments import parse_args
from main import run
from ray import tune


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


def monitor_dir_name(envname, config):
    """
    Returns a directory named for this config's hyperparams
    """
    return f'tmp-{envname}-{config["adv_force"]}'


def trainable(config):
    trial_dir = Path(tune.get_trial_dir()) if tune.get_trial_dir() is not None else Path.cwd()
    name = f'original-million-bucks_{config["adv_force"]}'
    args = parse_args([
        '--name', name,
        '--env', config["envname"],
        '--log',
        '--trainingconfig', str(Path(__file__).parent / 'trainingconfig.json'),
        '--root', str(trial_dir),
        '--adv_force', str(config["adv_force"]),
    ])
    args = add_config_args(args, config)
    args.monitor_dir = str(trial_dir / monitor_dir_name(config["envname"], config))
    logging.info(f'Running {name=} with {args=}')

    def evaluate(ts):
        # TODO may have to do something to prevent too-early stopping
        reward = get_mean_reward_last_n_steps(config["evaluate_mean_n"], args.monitor_dir)
        logging.info(f'{name} {reward=:.2f} {ts=}')
        tune.report(reward=reward)

    run(args, evaluate_fn=evaluate)


def main():
    num_samples = 10

    logging.basicConfig(level=logging.INFO)
    config = {
        "adv_force": tune.uniform(0, 1),  # TODO set range
        "envname": 'AdversarialAntBulletEnv-v0',
        "evaluate_mean_n": 1000,  # Number of timesteps over which to evaluate the mean reward
    }

    # TODO set search and sched
    search = HyperOptSearch()
    sched = ASHAScheduler()

    # Pass in a Trainable class or function to tune.run.
    local_dir = str(Path.cwd() / "ray")
    logging.info(f'{local_dir=}')
    anal = tune.run(trainable,
                    config=config,
                    num_samples=num_samples,
                    scheduler=sched,
                    search_alg=search,
                    local_dir=local_dir,
                    metric="reward",
                    mode="max",
                    log_to_file=True)
    logging.info(f'best config: {anal.best_config}')
    logging.info(f'best config: {anal.best_result}')


if __name__ == '__main__':
    main()
