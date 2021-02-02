import logging

from eval import report_results, config_logging, setup_log_dir
from main import run, parse_args

render = False
log = True
fixed_args = """
--evaluate
--N_eval_episodes=100
--env=AdversarialAntBulletEnv-v0
"""


def do(cmd_args):
    avg_reward, std_reward = run(parse_args(cmd_args))
    results = {
        "args": cmd_args,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
    }
    return results


def main():
    # Set up log dir
    if log:
        log_file, pickle_file = setup_log_dir()
        config_logging(log_file)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s'
        )

    # Run all combinations of hyperparameters
    results = []
    i = 0

    for adv_percentage in ['0.25', '0.5', '0.75', '1.0', None]:
        # for adv_percentage in [None]:
        for adv_percentage_rarl in ['0.25', '0.5', '0.75', '1.0']:
            # Do all percentages
            cmd_args = get_fixed_args()
            cmd_args.append(f'--name=original-big_{adv_percentage_rarl}')
            if adv_percentage is None:
                cmd_args.append(f'--force-no-adversarial')
            else:
                cmd_args.append(f'--force-adversarial')
                cmd_args.append(f'--adv_percentage={adv_percentage}')
            results.append(do(cmd_args))

            # Report results every so
            logging.info(f'{adv_percentage_rarl=} {adv_percentage=}')
            if log:
                report_results(pickle_file, results, iteration=i)
            i += 1

        # Do control
        cmd_args = get_fixed_args()
        cmd_args.append(f'--name=original-big')
        cmd_args.append('--control')
        if adv_percentage is None:
            cmd_args.append(f'--force-no-adversarial')
        else:
            cmd_args.append(f'--force-adversarial')
            cmd_args.append(f'--adv_percentage={adv_percentage}')
            cmd_args.append(f'--force-adv-name=original-big_{adv_percentage}')
        results.append(do(cmd_args))
        logging.info(f'control {adv_percentage=}')
        if log:
            report_results(pickle_file, results, iteration=i)
        i += 1

    if log:
        report_results(pickle_file, results)

    print_results(results)

    return results


def print_results(results):
    for r in results:
        def get_param(p): return next((a for a in r["args"] if p in a), '')

        print(get_param('--name'), get_param('--adv_percentage'), get_param('--force-adv-name'))
        print(r["avg_reward"], "+-", r["std_reward"])


def get_fixed_args():
    args_text = fixed_args
    args = args_text.split()
    if render:
        args.append('--render')
    return args


if __name__ == '__main__':
    finished_results = main()
