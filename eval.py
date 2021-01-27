import itertools
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from arguments import parse_args, set_args
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
hyperparameters = {
    "seed": [1, 2, 3, 4, 5],
    "adv_percentage": list(reversed([0.0, 0.25, 0.5, 0.75, 1.0])),
    "agent": ['', '--control'],
}


# Free params
# hyperparameters = {
#     "seed": [1],
#     "adv_percentage": list(reversed([0.0, 0.25, 0.5, 0.75, 1.0])),
#     "agent": ['', '--control'],
# }


# Enjoy one seed
# hyperparameters = {
#     "seed": [2],
#     "adv_percentage": [0.25],
#     "agent": ['--control'],
# }


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
        args = parse_args(cmd_args)
        set_args(args)
        avg_reward, std_reward = run(args)
        logging.info(f'reward={avg_reward}+={std_reward}')
        results.append({
            "hyperparameters": hp_set,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
        })

        # Report results every time we've gone through all agents for all adv_percentages
        if i + 1 % (len(hyperparameters["agent"]) * len(hyperparameters["adv_percentage"])) == 0:
            report_results(pickle_file, results, iteration=i)

    # Report final results
    report_results(pickle_file, results)


def report_results(pickle_file, results, iteration=None):
    """
    Summarize results and write to log and pickle, tagged with iteration number    """
    if iteration is None:
        iteration = 'FINAL'
    logging.info(f'{iteration=}')

    logging.info(f'{results=}')
    rarl_results = [r for r in results if r["hyperparameters"]["agent"] == '']
    rarl_avg_reward = get_avg(rarl_results, key="adv_percentage")
    control_results = [r for r in results if r["hyperparameters"]["agent"] == '--control']
    control_avg_reward = get_avg(control_results, key="adv_percentage")
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
