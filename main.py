import argparse
import json

from stable_baselines3 import PPO
# Set up environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import ProtagonistRarlEnv, AdversarialRarlEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_steps', type=int, default=None)  # Number of steps in a rolloout, N_traj in Algorithm 1
    parser.add_argument('--N_iter', type=int, default=None)
    parser.add_argument('--N_mu', type=int, default=None)
    parser.add_argument('--N_nu', type=int, default=None)
    parser.add_argument('--N_eval_episodes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--adv_percentage', type=float, default=None)
    parser.add_argument("--force-adversarial", action='store_true')
    parser.add_argument("--force-no-adversarial", action='store_true')
    arguments = parser.parse_args()

    arguments.logs = f'./logs/{arguments.name}'

    all_configs = json.load(open('trainingconfig.json'))
    assert not any('_' in config['name'] for config in all_configs)
    assert not any(any(k == 'name' for k in c['params'].keys()) for c in all_configs)

    if arguments.name:
        arguments.config_name = arguments.name
        if '_' in arguments.name:
            fields = arguments.name.split('_')
            assert len(fields) == 2
            arguments.config_name = fields[0]
            arguments.version = fields[1]
        for config in all_configs:
            if config['name'] == arguments.config_name:
                params = config['params']
                for k, v in params.items():
                    if arguments.__getattribute__(k):
                        print(f'config file overridden arguments[{k}] = {v}')
                    else:
                        print(f'config file set arguments[{k}] = {v}')
                        arguments.__setattr__(k, v)
                break
        arguments.pickle = f'./models/{arguments.name}'

    # Are we running RARL or control
    if arguments.adv_percentage is None:
        arguments.adv_percentage = 1.0

    if arguments.control:
        arguments.prot_name = 'control'
        arguments.adversarial = arguments.force_adversarial
    else:
        arguments.prot_name = 'prot'
        arguments.adversarial = not arguments.force_no_adversarial
    arguments.adv_name = 'adv'

    print(f'arguments: {arguments}')

    assert 0.0 <= arguments.adv_percentage <= 1.0
    assert arguments.N_steps % 2 == 0
    assert not (arguments.force_adversarial and arguments.force_no_adversarial)

    return arguments


def dummy(env_constructor, seed=None, evaluate_name=None):
    """
    Set up a dummy environment wrapper for Stable Baselines
    """
    env = make_vec_env(env_constructor, n_envs=1, seed=seed)
    # Automatically normalize the input features and reward
    if evaluate_name:
        env = VecNormalize.load(f'{evaluate_name}', env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                           clip_obs=10.)

    return env


def setup():
    bridge = Bridge()

    base_protenv = AdversarialCartPoleEnv(renders=args.render,
                                          adv_percentage=args.adv_percentage if args.adversarial else 0.0)
    prot_envname = f'{args.pickle}_{args.prot_name}env' if args.evaluate else None
    prot_env = dummy(lambda: ProtagonistRarlEnv(base_protenv, bridge), seed=args.seed,
                     evaluate_name=prot_envname)

    if args.adversarial:
        base_advenv = AdversarialCartPoleEnv(renders=args.render,
                                             adv_percentage=args.adv_percentage)
        adv_envname = f'{args.pickle}_{args.prot_name}env' if args.evaluate else None
        adv_env = dummy(lambda: AdversarialRarlEnv(base_advenv, bridge), seed=args.seed,
                        evaluate_name=adv_envname)
    else:
        adv_env = None

    if args.evaluate:
        prot_agent = PPO.load(f'{args.pickle}_{args.prot_name}')
        if prot_agent.seed != args.seed:
            print(f'warning: {prot_agent.seed=} does not match {args.seed=}')
        prot_agent.set_env(prot_env)

        if args.adversarial:
            adv_agent = PPO.load(f'{args.pickle}_{args.adv_name}')
            if adv_agent.seed != args.seed:
                print(f'warning: {adv_agent.seed=} does not match {args.seed=}')
            adv_agent.set_env(adv_env)
        else:
            adv_agent = None
    else:
        prot_logname = f'{args.logs}_{args.prot_name}' if args.logs else None
        prot_agent = PPO("MlpPolicy", prot_env, verbose=args.verbose, seed=args.seed,
                         tensorboard_log=prot_logname, n_steps=args.N_steps)

        if args.adversarial:
            adv_logname = f'{args.logs}_{args.adv_name}' if args.logs else None
            adv_agent = PPO("MlpPolicy", adv_env, verbose=args.verbose, seed=args.seed,
                            tensorboard_log=adv_logname, n_steps=args.N_steps)
        else:
            adv_agent = None

    bridge.link_agents(prot_agent, adv_agent)

    return prot_agent, adv_agent, prot_env, adv_env


args = get_args()


def main():
    prot, adv, prot_env, adv_env = setup()

    if not args.evaluate:
        # Train
        """
        Train according to Algorithm 1
        """
        for i in range(args.N_iter):
            # Do N_mu rollouts training the protagonist
            prot.learn(total_timesteps=args.N_mu * args.N_steps, reset_num_timesteps=i == 0)
            # Do N_nu rollouts training the adversary
            if adv is not None:
                adv.learn(total_timesteps=args.N_nu * args.N_steps, reset_num_timesteps=i == 0)

        prot.save(f'{args.pickle}_{args.prot_name}')
        prot_env.save(f'{args.pickle}_{args.prot_name}env')

        if adv is not None:
            adv.save(f'{args.pickle}_{args.adv_name}')
        if adv_env is not None:
            adv_env.save(f'{args.pickle}_{args.adv_name}env')

    prot_env.training = False
    prot_env.norm_reward = False
    avg_reward, std_reward = evaluate_policy(prot, prot_env, args.N_eval_episodes)
    print(f'{avg_reward=}')

    prot_env.close()
    if adv_env is not None:
        adv_env.close()


if __name__ == '__main__':
    main()
