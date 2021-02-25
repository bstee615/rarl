import logging

import numpy as np

from eval import log, setup_log_dir, config_logging, get_fixed_args, do, report_results

env = 'AdversarialHalfCheetahBulletEnv-v0'


def main():
    # Set up log dir
    if log:
        log_file, pickle_file = setup_log_dir(eval_name=f'mass-friction-{env}')
        config_logging(log_file)
    else:
        pickle_file = None
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s'
        )

    # Run all combinations of hyperparameters
    results = []
    i = 0
    for mass_friction in ['mass', 'friction']:
        for percentage in np.arange(0.5, 1.5, 0.1):
            for is_rarl in [True, False]:
                cmd_args = get_fixed_args(is_rarl, env)
                # Eval specific params
                cmd_args.append(f'--force-no-adversarial')
                cmd_args.append(f'--{mass_friction}_percentage={percentage}')
                result = do(cmd_args)
                results.append(result)

                # Report results every so
                logging.info(f'{is_rarl=} {mass_friction=} {percentage}')
                if log:
                    report_results(pickle_file, results, iteration=i)
                i += 1

    return results


if __name__ == '__main__':
    finished_results = main()
