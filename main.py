import logging
from functools import partial

import numpy as np
from stable_baselines3 import PPO
# Set up environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from arguments import args, parse_args
from bridge import Bridge
from gym_rarl.envs.rarl_env import ProtagonistRarlEnv, AdversarialRarlEnv


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
    if seed:
        env.seed(seed)

    return env


def setup():
    bridge = Bridge()

    loaded_constructor = partial(args.env_constructor,
                                 render=args.render,
                                 adv_percentage=args.adv_percentage,
                                 mass_percentage=args.mass_percentage,
                                 friction_percentage=args.friction_percentage,
                                 )

    base_protenv = loaded_constructor(
        render=args.render,
        adv_percentage=args.adv_percentage)
    prot_env = dummy(lambda: ProtagonistRarlEnv(base_protenv, bridge), seed=args.seed,
                     evaluate_name=f'{args.pickle}-{args.prot_envname}' if args.evaluate else None)

    if args.adversarial:
        base_advenv = loaded_constructor(
            render=args.render,
            adv_percentage=args.adv_percentage
        )
        adv_env = dummy(lambda: AdversarialRarlEnv(base_advenv, bridge), seed=args.seed,
                        evaluate_name=f'{args.pickle}-{args.adv_envname}' if args.evaluate else None)
    else:
        adv_env = None

    if args.evaluate:
        prot_agent = PPO.load(f'{args.pickle}-{args.prot_name}')
        if prot_agent.seed != args.seed:
            logging.info(f'warning: {prot_agent.seed=} does not match { args.seed=}')
        prot_agent.set_env(prot_env)

        if args.adversarial:
            adv_agent = PPO.load(f'{args.pickle}-{args.adv_name}')
            if adv_agent.seed != args.seed:
                logging.info(f'warning: {adv_agent.seed=} does not match { args.seed=}')
            adv_agent.set_env(adv_env)
        else:
            adv_agent = None
    else:
        prot_logname = f'{args.logs}-{args.prot_name}' if args.logs else None
        prot_agent = PPO("MlpPolicy", prot_env, verbose=args.verbose, seed=args.seed,
                         tensorboard_log=prot_logname, n_steps=args.N_steps)

        if args.adversarial:
            adv_logname = f'{args.logs}-{args.adv_name}' if args.logs else None
            adv_agent = PPO("MlpPolicy", adv_env, verbose=args.verbose, seed=args.seed,
                            tensorboard_log=adv_logname, n_steps=args.N_steps)
        else:
            adv_agent = None

    bridge.link_agents(prot_agent, adv_agent)

    return prot_agent, adv_agent, prot_env, adv_env


def run():
    prot, adv, prot_env, adv_env = setup()
    try:
        if args.evaluate:
            prot_env.training = False
            prot_env.norm_reward = False
            reward, lengths = evaluate_policy(prot, prot_env, args.N_eval_episodes,
                                              max_episode_length=args.N_eval_timesteps,
                                              return_episode_rewards=True)
            avg_reward = np.mean(reward)
            std_reward = np.std(reward)
            return avg_reward, std_reward
        else:
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

            prot.save(f'{args.pickle}-{args.prot_name}')
            prot_env.save(f'{args.pickle}-{args.prot_envname}')

            if adv is not None:
                adv.save(f'{args.pickle}-{args.adv_name}')
            if adv_env is not None:
                adv_env.save(f'{args.pickle}-{args.adv_envname}')
    finally:
        prot_env.close()
        if adv_env is not None:
            adv_env.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    result = run()
    if result is not None:
        avg_reward, std_reward = result
        logging.info(f'reward={avg_reward}+={std_reward}')
