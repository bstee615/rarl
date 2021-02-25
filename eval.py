import itertools
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from arguments import parse_args
from main import run

render = True
args_fmt = """
--evaluate --force-adversarial
--N_eval_episodes=10 --N_eval_timesteps=1000
--name=original_{1}-{0} --seed={0}
--adv_percentage={1}
--env=AdversarialAntBulletEnv-v0
{2}
"""

# Run 5 seeds for 5 different adv_percentages
# hyperparameters = {
#     "seed": [1, 2, 3, 4, 5],
#     "adv_percentage": list(reversed([0.0, 0.25, 0.5, 0.75, 1.0])),
#     "agent": ['', '--control'],
# }


# Free params
# hyperparameters = {
#     "seed": [1],
#     "adv_percentage": list(reversed([0.0, 0.25, 0.5, 0.75, 1.0])),
#     "agent": ['', '--control'],
# }


# Enjoy one seed
hyperparameters = {
    "seed": [12345],
    "adv_percentage": [0.5, 0.75, ''],
    "agent": ['', '--control'],
}


def get_avg(results, key):
    """
    Return the average reward over one hyperparameter key from a list of runs
    """
    num_percentages = len(set(k["hyperparameters"]["seed"] for k in results))
    sum_reward = defaultdict(float)
    for r in results:
        sum_reward[r["hyperparameters"][key]] += r["avg_reward"]
    avg_reward = {percentage: r / num_percentages for percentage, r in sum_reward.items()}
    return avg_reward


def main():
    # Set up log dir
    log_file, pickle_file = setup_log_dir()
    config_logging(log_file)

    # Run all combinations of hyperparameters
    all_hp_combinations = [
        {label: hp for label, hp in zip(hyperparameters.keys(), p)}
        for p in itertools.product(*hyperparameters.values())
    ]
    results = []
    for i, hp_set in enumerate(all_hp_combinations):
        logging.info(f'{hp_set=}')
        cmd_args = args_fmt.format(*hp_set.values()).split()
        if render:
            cmd_args.append('--render')
        do(cmd_args)

        # Report results every time we've gone through all agents for all adv_percentages
        if i + 1 % (len(hyperparameters["agent"]) * len(hyperparameters["adv_percentage"])) == 0:
            report_results(pickle_file, results, iteration=i)

    # Report final results
    report_results(pickle_file, results)


def setup_log_dir():
    all_logs_dir = Path('logs')
    filename_count = 0
    while (log_dir := all_logs_dir / f'eval-{filename_count}').is_dir():
        filename_count += 1
    log_dir.mkdir()
    log_file = log_dir / 'info.log'
    pickle_file = log_dir / 'summary.pkl'
    return log_file, pickle_file


def config_logging(log_file):
    file_handler = logging.FileHandler(filename=str(log_file))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=handlers
    )


def report_results(pickle_file, results, iteration=None):
    """
    Summarize results and write to log and pickle, tagged with iteration number    """
    if iteration is None:
        iteration = 'FINAL'
    logging.info(f'{iteration=}')
    logging.info(f'{results=}')
    if pickle_file is not None:
        with pickle_file.open('wb') as f:
            pickle.dump(results, f)


def do(cmd_args):
    args = parse_args(cmd_args)
    rew = run(args)
    import numpy as np
    avg_reward, std_reward = np.mean(rew), np.std(rew)
    logging.info(f'reward={avg_reward}+={std_reward}')


if __name__ == '__main__':
    basic = '--render --evaluate --N_eval_episodes=5 --env AdversarialAntBulletEnv-v0'
    cmds = []
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
    )
    cmds.append(basic + ' --name original-big_0.5 --force-no-adversarial')
    cmds.append(basic + ' --name original-big --control')
    for p in ['0.25', '0.5', '0.75', '1.0']:
        cmds.append(basic + ' --name original-big_0.5' + f' --adv_percentage {p}' + ' --force-adversarial')
        cmds.append(
            basic + ' --name original-big --control' + f' --adv_percentage {p} --force-adv-name original-big_{p}' + ' --force-adversarial')
    for cmd_args in cmds:
        do(cmd_args.split())
    # main()
