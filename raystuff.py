import logging
from pathlib import Path

import numpy as np
from ray.tune.schedulers import ASHAScheduler
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


def monitor_dir_name(envname, adv_force):
    """
    Returns a directory named for this config's hyperparams
    """
    return f'tmp-{envname}-{adv_force}'


def trainable(config, name_fmt, envname, trainingconfig, evaluate_mean_n):
    # Parse arguments
    trial_dir = Path(tune.get_trial_dir()) if tune.get_trial_dir() is not None else Path.cwd()
    adv_force = config["adv_force"]
    name = name_fmt.format(adv_force=adv_force)
    cmd_args = [
        '--name', name,
        '--env', envname,
        '--log',
        '--trainingconfig', str(trainingconfig),
        '--root', str(trial_dir),
        '--monitor-dir', str(monitor_dir_name(envname, adv_force))
    ]
    cmd_args += ['--adv_force', str(adv_force)]
    args = parse_args(cmd_args)
    # Add adversarial force
    logging.info(f'Running {name=} with {args=}')

    def evaluate(ts):
        reward = get_mean_reward_last_n_steps(evaluate_mean_n, args.monitor_dir)
        logging.info(f'{name} {reward=:.2f} {ts=}')
        tune.report(reward=reward)

    run(args, evaluate_fn=evaluate)


def main():
    logging.basicConfig(level=logging.INFO)

    # Raylib parameters
    num_samples = 100
    envname = 'AdversarialAntBulletEnv-v0'
    trainingconfig = Path.cwd() / 'trainingconfig.json'
    evaluate_mean_n = 1000  # Number of timesteps over which to evaluate the mean reward
    name_fmt = 'little_{adv_force}'

    config = {
        # TODO: sample from control once, then different adversarial strengths
        # Range is centered on the force that achieves the closest reward to the control (7.5)
        "adv_force": tune.randn(7.5, 2.5),
    }

    # https://docs.ray.io/en/master/tune/tutorials/overview.html#which-search-algorithm-scheduler-should-i-choose
    # Use BOHB for larger problems with a small number of hyperparameters
    # search = TuneBOHB(max_concurrent=4, metric="mean_loss", mode="min")
    # sched = HyperBandForBOHB(
    #     time_attr="training_iteration",
    #     max_t=100,
    # )

    # Implicitly use random search if search algo is not specified
    sched = ASHAScheduler(
        time_attr='training_iteration',
        max_t=100,
        grace_period=10,  # Unit is iterations, not timesteps. TODO configure
    )

    # Pass in a Trainable class or function to tune.run.
    local_dir = str(Path.cwd() / "ray")
    logging.info(f'{local_dir=}')
    anal = tune.run(tune.with_parameters(trainable,
                                         envname=envname,
                                         trainingconfig=trainingconfig,
                                         evaluate_mean_n=evaluate_mean_n,
                                         name_fmt=name_fmt),
                    config=config,
                    num_samples=num_samples,
                    scheduler=sched,
                    local_dir=local_dir,
                    metric="reward",
                    mode="max",
                    log_to_file=True)
    logging.info(f'best config: {anal.best_config}')
    logging.info(f'best config: {anal.best_result}')


if __name__ == '__main__':
    main()
