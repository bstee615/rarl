import logging

from eval import setup_log_dir, config_logging, report_results
from main import run, parse_args

render = False
fixed_args = """
--evaluate --force-adversarial
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
    log_file, pickle_file = setup_log_dir()
    config_logging(log_file)

    # Run all combinations of hyperparameters
    results = []
    i = 0

    for adv_percentage in ['0.25', '0.5', '0.75', '1.0']:
        for adv_percentage_name in ['0.25', '0.5', '0.75', '1.0']:
            # Do all percentages
            cmd_args = get_fixed_args()
            cmd_args.append(f'--name=original-big_{adv_percentage_name}')
            cmd_args.append(f'--adv_percentage={adv_percentage}')
            results.append(do(cmd_args))

            # Report results every so
            logging.info(f'{adv_percentage_name=} {adv_percentage=}')
            report_results(pickle_file, results, iteration=i)
            i += 1

        # Do control
        cmd_args = get_fixed_args()
        cmd_args.append(f'--name=original-big')
        cmd_args.append('--control')
        cmd_args.append(f'--force-adversarial')
        cmd_args.append(f'--adv_percentage={adv_percentage}')
        cmd_args.append(f'--force-adv-name=original-big_{adv_percentage}')
        results.append(do(cmd_args))
        logging.info(f'control {adv_percentage=}')
        report_results(pickle_file, results, iteration=i)
        i += 1

    report_results(pickle_file, results)


def get_fixed_args():
    args_text = fixed_args
    args = args_text.split()
    if render:
        args.append('--render')
    return args


if __name__ == '__main__':
    main()
