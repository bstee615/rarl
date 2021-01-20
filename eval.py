import itertools
import logging
import pickle
from collections import defaultdict
from pathlib import Path

from main import *


def get_avg(results):
    num_percentages = len(set(k["hyperparameters"]["seed"] for k in results))
    sum_reward = defaultdict(float)
    for r in results:
        reward = r["avg_reward"]
        sum_reward[r["hyperparameters"]["adv_percentage"]] += reward
    avg_reward = {percentage: r / num_percentages for percentage, r in sum_reward.items()}
    return avg_reward


def main():
    render = True
    args_fmt = \
        "--name=original_{0} --seed={0} --evaluate --force-adversarial --N_eval_episodes=10 --adv_percentage={1} {2}"

    # Set up log dir
    all_logs_dir = Path('logs')
    filename_count = 0
    while (log_dir := all_logs_dir / f'eval-{filename_count}').is_dir():
        filename_count += 1
    log_dir.mkdir()
    log_file = log_dir / 'info.log'
    pickle_file = log_dir / 'summary.pkl'
    file_handler = logging.FileHandler(filename=str(log_file))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=handlers
    )

    # Run 5 seeds for 5 different adv_percentages
    hyperparameters = {
        "seed": [1, 2, 3, 4, 5],
        "adv_percentage": reversed([0.0, 0.25, 0.5, 0.75, 1.0]),
        "agent": ['', '--control'],
    }

    # Enjoy one seed
    # hyperparameters = {
    #     "seed": [2],
    #     "adv_percentage": [0.0],
    #     "agent": ['', '--control'],
    # }
    all_hp_combinations = [
        {label: hp for label, hp in zip(hyperparameters.keys(), p)}
        for p in itertools.product(*hyperparameters.values())
    ]

    results = []
    args_sets = []
    for hp_set in all_hp_combinations:
        logging.info(f'{hp_set=}')
        cmd_args = args_fmt.format(*hp_set.values()).split()
        if render:
            cmd_args.append('--render')
        args = get_args(cmd_args)
        avg_reward, std_reward = run(args)
        logging.info(f'{avg_reward=}, {std_reward=}')
        results.append({
            "hyperparameters": hp_set,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
        })

    logging.info(f'{results=}')

    rarl_results = [r for r in results if r["hyperparameters"]["agent"] == '']
    rarl_avg_reward = get_avg(rarl_results)

    control_results = [r for r in results if r["hyperparameters"]["agent"] == '--control']
    control_avg_reward = get_avg(control_results)

    logging.info(f'{rarl_avg_reward=}')
    logging.info(f'{control_avg_reward=}')

    with pickle_file.open('wb') as f:
        summary = {
            "rarl": {
                "results": rarl_results,
                "avg_reward": rarl_avg_reward,
            },
            "control": {
                "results": control_results,
                "avg_reward": control_avg_reward,
            },
        }
        logging.info(f'{summary=}')
        pickle.dump(summary, f)


if __name__ == '__main__':
    main()
