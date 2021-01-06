import argparse
import json

from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines3 import PPO
# Set up environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import MainRarlEnv, AdversarialRarlEnv


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
        arguments.pickle = f'./models/{arguments.config_name}'

    assert arguments.N_steps % 2 == 0

    print(f'arguments: {arguments}')

    assert arguments.N_steps % 2 == 0

    return arguments


args = get_args()


def dummy(env_constructor, seed):
    """
    Set up a dummy environment wrapper for Stable Baselines
    """
    env = make_vec_env(env_constructor, seed=seed)
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=10.)
    return env


def setup_adv():
    """
    Setup models and env for adversarial training
    """
    base_env = AdversarialCartPoleEnv(renders=args.render)
    if args.seed:
        base_env.seed(args.seed)

    if args.evaluate:
        prot_agent = PPO.load(f'{args.pickle}_prot')
        adv_agent = PPO.load(f'{args.pickle}_prot')
        if prot_agent.seed != args.seed:
            print(f'warning: {prot_agent.seed=} does not match {args.seed=}')
        if adv_agent.seed != args.seed:
            print(f'warning: {adv_agent.seed=} does not match {args.seed=}')

    bridge = Bridge()
    main_env = dummy(lambda: MainRarlEnv(base_env, bridge), seed=args.seed)
    adv_env = dummy(lambda: AdversarialRarlEnv(base_env, bridge), seed=args.seed)

    # Set up agents
    if not args.evaluate:
        prot_agent = PPO("MlpPolicy", main_env, verbose=args.verbose, seed=args.seed,
                         tensorboard_log=f'{args.logs}_prot' if args.logs else None, n_steps=args.N_steps)
        adv_agent = PPO("MlpPolicy", adv_env, verbose=args.verbose, seed=args.seed,
                        tensorboard_log=f'{args.logs}_adv' if args.logs else None, n_steps=args.N_steps)

    # Link agents
    bridge.link_agents(prot_agent, adv_agent)

    return prot_agent, adv_agent, main_env, adv_env


def setup_control():
    """
    Setup a normal model and environment a a control
    """
    if args.evaluate:
        model = PPO.load(f'{args.pickle}_control')
        if model.seed != args.seed:
            print(f'warning: {model.seed=} does not match {args.seed=}')

    env = dummy(lambda: CartPoleBulletEnv(renders=args.render), seed=args.seed)

    if not args.evaluate:
        model = PPO("MlpPolicy", env, verbose=args.verbose, seed=args.seed,
                    tensorboard_log=f'{args.logs}_control' if args.logs else None, n_steps=args.N_steps)
    return model, env


def main():
    if args.control:
        model, env = setup_control()
        if args.evaluate:
            avg_reward, std_reward = evaluate_policy(model, env, args.N_eval_episodes)
            print(f'{avg_reward=}')
        else:
            model.learn(total_timesteps=args.N_iter * args.N_mu * args.N_steps)
            model.save(f'{args.pickle}_control')
        env.close()
    else:
        prot, adv, prot_env, adv_env = setup_adv()
        if args.evaluate:
            avg_reward, std_reward = evaluate_policy(prot, prot_env, args.N_eval_episodes)
            print(f'{avg_reward=}')
        else:
            """
            Train according to Algorithm 1
            """
            for i in range(args.N_iter):
                # Do N_mu rollouts training the protagonist
                prot.learn(total_timesteps=args.N_mu * args.N_steps, reset_num_timesteps=False)
                # Do N_nu rollouts training the adversary
                adv.learn(total_timesteps=args.N_nu * args.N_steps, reset_num_timesteps=False)
            prot.save(f'{args.pickle}_prot')
            adv.save(f'{args.pickle}_adv')
        prot_env.close()
        adv_env.close()


if __name__ == '__main__':
    main()
