import logging

from eval import log, setup_log_dir, config_logging, get_fixed_args, do, report_results

env = 'AdversarialHalfCheetahBulletEnv-v0'


def main():
    # Set up log dir
    if log:
        log_file, pickle_file = setup_log_dir(eval_name=f'{env}')
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
    for adv_percentage in ['1.0', None]:
        for is_rarl in [True, False]:
            cmd_args = get_fixed_args(is_rarl, env)
            # Eval specific params
            if adv_percentage is None:
                cmd_args.append(f'--force-no-adversarial')
            else:
                cmd_args.append(f'--force-adversarial')
                cmd_args.append(f'--adv_percentage={adv_percentage}')
                # TODO: Consider the effect of suboptimal normalization.
                #  This adversary will not perform like it would
                #  in an environment which it is accustomed to (normalized correctly).
                #  So the control will be perturbed suboptimally, may perform artificially better.
                #  Effect is gone in environmental condition experiment.
                adv_suffix = '_rarl'
                cmd_args.append(f'--force-adv-name=original-big{adv_suffix}')
            result = do(cmd_args)
            results.append(result)

            # Report results every so
            logging.info(f'{is_rarl=} {adv_percentage=}')
            if log:
                report_results(pickle_file, results, iteration=i)
            i += 1

    return results


if __name__ == '__main__':
    finished_results = main()
