import json
import logging
from pathlib import Path

import numpy as np
from ray.tune.schedulers import ASHAScheduler
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from arguments import parse_args
from eval import do, get_fixed_args
from main import run
from ray import tune


def get_mean_reward_last_n_steps(n, monitor_dir):
    """
    Get the mean episodal reward recorded over the last n steps
    """
    x, y = ts2xy(load_results(monitor_dir), 'timesteps')
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


def eval_robustness(args, prot, env, trainingconfig, name):
    """
    Evaluate robustness to different environment parameters
    """
    prot_pickle = Path(args.prot_pickle)
    env_pickle = Path(args.env_pickle)
    name_dir = prot_pickle.parent
    tmp_dir = name_dir.parent / (name_dir.name + '_tmp')
    tmp_dir.mkdir(exist_ok=True)
    tmp_prot_pickle = tmp_dir / prot_pickle.name
    tmp_env_pickle = tmp_dir / env_pickle.name
    prot.save(str(tmp_prot_pickle))
    prot.get_env().save(str(tmp_env_pickle))
    results = []
    logging.info(f'Evaluating robustness of {name=}')
    for percentage in ['0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4']:
        cmd_args = get_fixed_args(True, env, N_eval_episodes=10, name=f'--name={name}')
        # Eval specific params
        cmd_args.append(f'--force-no-adversarial')
        cmd_args.append(f'--mass_percentage={percentage}')
        cmd_args.append(f'--trainingconfig={trainingconfig}')
        cmd_args.append(f'--root={args.root}')
        cmd_args.append(f'--force-prot-name={tmp_dir.name}')
        result = do(cmd_args)
        results.append(result)
    return np.average(np.array([result["avg_reward"] for result in results]))


def trainable(config, name_fmt, envname, trainingconfig, baseline):
    # Parse arguments
    trial_dir = Path(tune.get_trial_dir())
    adv_force = config["adv_force"]
    name = name_fmt.format(adv_force=adv_force)
    cmd_args = [
        '--name', name,
        '--env', envname,
        '--trainingconfig', str(trainingconfig),
        '--root', str(trial_dir),
        '--monitor-dir', str(monitor_dir_name(envname, adv_force))
    ]
    cmd_args += ['--adv_force', str(adv_force)]
    args = parse_args(cmd_args)
    if baseline is not None:
        assert len(baseline) == args.N_iter
    # Add adversarial force
    logging.info(f'Running {name=} with {args=}')

    def evaluate(prot, ts):
        report = {}
        report["reward"] = get_mean_reward_last_n_steps(args.N_mu * args.N_steps, args.monitor_dir)
        report["robustness"] = eval_robustness(args, prot, envname, trainingconfig, name)
        if baseline is not None:
            report["robustness_vs_baseline"] = baseline[ts]["robustness"] - report["robustness"]
        tune.report(**report)

    run(args, evaluate_fn=evaluate)


def record_baseline(baseline_dir, baseline_logname, name, envname, trainingconfig):
    cmd_args = [
        '--name', name,
        '--env', envname,
        '--verbose',
        '--log',
        '--trainingconfig', str(trainingconfig),
        '--root', str(baseline_dir),
        '--monitor-dir', str(monitor_dir_name(envname, 'baseline'))
    ]
    args = parse_args(cmd_args)

    baseline = []
    baseline_file = baseline_dir / baseline_logname
    with baseline_file.open('w') as f:
        json.dump(baseline, f)  # Zonk it to show we can write to it

    def evaluate(prot, ts):
        myeval = {}
        reward = get_mean_reward_last_n_steps(args.N_mu * args.N_steps, args.monitor_dir)
        logging.info(f'baseline {reward=:.2f} {ts=}')
        robustness = eval_robustness(args, prot, envname, trainingconfig, name)
        logging.info(f'baseline {robustness=:.2f} {ts=}')

        myeval["reward"] = reward
        myeval["robustness"] = robustness

        baseline.append(myeval)

    logging.info(f'Running baseline {name=} with {args=}')
    run(args, evaluate_fn=evaluate)

    with baseline_file.open('w') as f:
        json.dump(baseline, f)


def main():
    logging.basicConfig(level=logging.INFO)

    # Raylib parameters
    num_samples = 10
    envname = 'AdversarialAntBulletEnv-v0'
    trainingconfig = Path.cwd() / 'trainingconfig.json'
    name = 'big'
    name_fmt = name + '_{adv_force}'
    max_t = 500

    # Baseline parameters
    baseline_dir = Path.cwd() / 'ray/baseline'
    baseline_logname = f'baseline_{name}-{envname}.json'

    # Config
    should_record_baseline = False
    metric = "robustness"

    if should_record_baseline:
        # Run baseline and exit
        baseline_dir.mkdir(parents=True, exist_ok=True)
        record_baseline(baseline_dir, baseline_logname, name, envname, trainingconfig)
        exit(0)
    else:
        # Load baseline
        baseline_file = baseline_dir / baseline_logname
        if baseline_file.exists():
            with baseline_file.open() as f:
                baseline = json.load(f)
            if len(baseline) != max_t:
                logging.warning(f'Baseline at {baseline_file} is not complete. Skipping...')
                baseline = None
        else:
            baseline = None

    config = {
        # Range is centered on the force that achieves the closest reward to the control (7.5)
        "adv_force": tune.qrandn(7.5, 2.5, 0.1),
    }

    # https://docs.ray.io/en/master/tune/tutorials/overview.html#which-search-algorithm-scheduler-should-i-choose
    # Use BOHB for larger problems with a small number of hyperparameters
    # search = TuneBOHB(max_concurrent=4, metric="mean_loss", mode="min")
    # sched = HyperBandForBOHB(
    #     time_attr="training_iteration",
    #     max_t=100,
    # )

    # Implicitly use random search if search algo is not specified
    # Training config "big"
    sched = ASHAScheduler(
        time_attr='training_iteration',  # Unit t is iterations, not timesteps.
        max_t=max_t,
        grace_period=125,
        # This is configured to start evaluating at the point where agent performance starts to diverge wrt adv strength
    )

    # Pass in a Trainable class or function to tune.run.
    local_dir = str(Path.cwd() / "ray")
    logging.info(f'{local_dir=}')
    anal = tune.run(tune.with_parameters(trainable,
                                         envname=envname,
                                         trainingconfig=trainingconfig,
                                         name_fmt=name_fmt,
                                         baseline=baseline),
                    config=config,
                    num_samples=num_samples,
                    scheduler=sched,
                    local_dir=local_dir,
                    metric=metric,
                    mode="max",
                    log_to_file=True)
    logging.info(f'best config: {anal.best_config}')
    logging.info(f'best config: {anal.best_result}')


if __name__ == '__main__':
    main()
