import argparse
import json
import logging
import sys

from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.adv_walkers import AdversarialHopperEnv

args = None


def parse_args(cmd_args=None):
    if cmd_args is None:
        cmd_args = sys.argv[1:]

    # Parse name to know which config to do
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, required=True)
    name_arguments, remaining_args = parser.parse_known_args(cmd_args)

    # Load config file
    config_arguments = get_config_arguments(name_arguments)

    # Parse other commandline options
    # Hyperparameters
    parser.add_argument('--N_steps', type=int)  # Number of steps in a rolloout, N_traj in Algorithm 1
    parser.add_argument('--N_iter', type=int)
    parser.add_argument('--N_mu', type=int)
    parser.add_argument('--N_nu', type=int)
    parser.add_argument('--N_eval_episodes', type=int)
    parser.add_argument('--N_eval_timesteps', type=int)
    parser.add_argument('--adv_percentage', type=float)
    parser.add_argument('--seed', type=int)
    # The name of the adversarial environment class
    parser.add_argument("--env", type=str, default='AdversarialCartPoleEnv',
                        help=', '.join([str(e) for e in [AdversarialCartPoleEnv, AdversarialHopperEnv]]))
    # Flags
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--force-adversarial", action='store_true')
    parser.add_argument("--force-no-adversarial", action='store_true')

    arguments = parser.parse_args(cmd_args, namespace=config_arguments)

    # Set extra variables
    populate_derivatives(arguments)

    logging.info(f'All arguments: {arguments}')

    validate_arguments(arguments)

    return arguments


def validate_arguments(arguments):
    """
    List of sanity assertions for commandline and trainingconfig arguments
    """
    if arguments.adv_percentage:
        assert 0.0 <= arguments.adv_percentage
    if arguments.N_steps:
        assert arguments.N_steps % 2 == 0
    assert not (arguments.force_adversarial and arguments.force_no_adversarial)


def populate_derivatives(arguments):
    """
    Add derivative arguments from the already parsed ones.
    """
    arguments.pickle = f'./models/{arguments.name}'
    arguments.logs = f'./logs/{arguments.name}'
    # Are we running RARL or control
    if arguments.adv_percentage is None:
        arguments.adv_percentage = 1.0
    arguments.env_constructor = globals()[arguments.env]
    if arguments.control:
        arguments.prot_name = f'control-{arguments.env}'
        arguments.adversarial = arguments.force_adversarial
    else:
        arguments.prot_name = f'prot-{arguments.env}'
        arguments.adversarial = not arguments.force_no_adversarial
    arguments.adv_name = f'adv-{arguments.env}'


def get_config_arguments(existing_arguments):
    """
    Parse config arguments given --name.
    The config name must be a valid filepath.
    It could be simple or compound, where --name="name_version" uses config "name"
    and stores logs and models with prefix "name_version".
    Returns a Namespace object with parameters from trainingconfig.
    """
    all_configs = json.load(open('trainingconfig.json'))
    assert not any('_' in config['name'] for config in all_configs)
    assert not any(any(k == 'name' for k in c['params'].keys()) for c in all_configs)
    configfile_arguments = argparse.Namespace()
    config_name = existing_arguments.name
    if '_' in existing_arguments.name:
        fields = existing_arguments.name.split('_')
        assert len(fields) == 2
        config_name = fields[0]
    my_config = next(c for c in all_configs if c['name'] == config_name)
    params = my_config['params']
    for k, v in params.items():
        logging.info(f'config file set arguments[{k}] = {v}')
        configfile_arguments.__setattr__(k, v)
    return configfile_arguments


def set_args(new_args):
    global args
    args = new_args