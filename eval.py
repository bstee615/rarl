import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from main import run, parse_args

enjoy = False
log = True
simple_reward = False


def get_fixed_args(is_rarl, env):
    fixed_args = f"""
    --evaluate
    --N_eval_episodes={3 if enjoy else 100}
    {'--simple-reward' if simple_reward else ''}
    --env={env}
    """
    args_text = fixed_args
    args = args_text.split()
    render = enjoy
    if render:
        args.append('--render')
    if is_rarl:
        rarl_suffix = '_rarl'
    else:
        rarl_suffix = ''
        args.append(f'--control')
    args.append(f'--name=original-big{rarl_suffix}')
    return args


def setup_log_dir(eval_name=None):
    all_logs_dir = Path('logs')
    if eval_name:
        log_dir = all_logs_dir / f'eval-{eval_name}'
    else:
        filename_count = 0
        while (log_dir := all_logs_dir / f'eval-{filename_count}').is_dir():
            filename_count += 1
    assert not log_dir.is_dir()
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


def do(cmd_args):
    args = parse_args(cmd_args)
    rewards = run(args)
    avg_reward = np.average(rewards)
    std_reward = np.std(rewards)
    results = {
        "args": cmd_args,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "rewards": rewards,
    }
    return results


def report_results(pickle_file, results, iteration=None):
    """
    Summarize results and write to log and pickle, tagged with iteration number
    """
    if iteration is None:
        iteration = 'FINAL'
    else:
        logging.info(f'{iteration=}')
        logging.info(f'{results[iteration]["args"]=} {results[iteration]["avg_reward"]=}')
    # logging.info(f'{results=}')
    if pickle_file is not None:
        with pickle_file.open('wb') as f:
            pickle.dump(results, f)
