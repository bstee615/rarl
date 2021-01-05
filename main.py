import argparse

from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines3 import PPO
# Set up environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import MainRarlEnv, AdversarialRarlEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_steps', type=int, default=64)
    parser.add_argument('--N_iter', type=int, default=10)
    parser.add_argument('--N_mu', type=int, default=2)
    parser.add_argument('--N_nu', type=int, default=2)
    parser.add_argument('--N_traj', type=int, default=128)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--demo', dest='demo_mode', action='store_true')
    parser.add_argument('--name', type=str, default='rarl-temp')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--control', action='store_true')
    arguments = parser.parse_args()

    assert arguments.N_steps % 2 == 0
    # assert arguments.N_traj % arguments.N_steps == 0
    # arguments.N_traj_over_n_steps = arguments.N_traj / arguments.N_steps

    arguments.pickle = f'./models/{arguments.name}'
    arguments.logs = f'./logs/{arguments.name}'

    print(f'pickling to {arguments.pickle}')
    print(f'logging to {arguments.logs}')

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
    base_env = AdversarialCartPoleEnv(renders=args.demo_mode)
    # base_env.seed(args.seed)
    base_env.seed(100)

    bridge = Bridge()
    main_env = dummy(lambda: MainRarlEnv(base_env, bridge), seed=args.seed)
    adv_env = dummy(lambda: AdversarialRarlEnv(base_env, bridge), seed=args.seed)
    # main_env.seed(args.seed)
    # adv_env.seed(args.seed)
    main_env.seed(100)
    adv_env.seed(100)

    # Set up agents
    if args.log:
        prot_agent = PPO("MlpPolicy", main_env, verbose=args.verbose, seed=args.seed,
                         tensorboard_log=f'{args.logs}_prot', n_steps=args.N_steps)
        adv_agent = PPO("MlpPolicy", adv_env, verbose=args.verbose, seed=args.seed,
                        tensorboard_log=f'{args.logs}_adv', n_steps=args.N_steps)
    else:
        prot_agent = PPO("MlpPolicy", main_env, verbose=args.verbose, seed=args.seed, n_steps=args.N_steps)
        adv_agent = PPO("MlpPolicy", adv_env, verbose=args.verbose, seed=args.seed, n_steps=args.N_steps)

    # Link agents
    bridge.link_agents(prot_agent, adv_agent)

    env = main_env
    return prot_agent, adv_agent, env


def setup_control():
    """
    Setup a normal model and environment a a control
    """
    env = dummy(lambda: CartPoleBulletEnv(renders=args.demo_mode), seed=args.seed)
    # env.seed(args.seed)
    env.seed(100)
    if args.log:
        model = PPO("MlpPolicy", env, verbose=args.verbose, seed=args.seed, tensorboard_log=f'{args.logs}_control',
                    n_steps=args.N_steps)
    else:
        model = PPO("MlpPolicy", env, verbose=args.verbose, seed=args.seed,
                    n_steps=args.N_steps)
    return model, env


def train(prot, adv):
    """
    Train according to Algorithm 1
    """
    reset_prot = True
    reset_adv = True
    for i in range(args.N_iter):
        for j in range(args.N_mu):
            # rollout N_traj timesteps training the protagonist
            prot.learn(total_timesteps=args.N_traj, reset_num_timesteps=reset_prot)
            reset_prot = False
        for j in range(args.N_nu):
            # rollout N_traj timesteps training the adversary
            adv.learn(total_timesteps=args.N_traj, reset_num_timesteps=reset_adv)
            reset_adv = False
    prot.save(f'{args.pickle}_prot')
    adv.save(f'{args.pickle}_adv')
    del prot


def run(env):
    """
    Run model in env
    """
    model = PPO.load(f'{args.pickle}_prot')
    obs = env.reset()
    for ts in range(10000):
        if args.demo_mode:
            env.render()
        action, _ = model.predict(obs, deterministic=args.seed is not None)
        obs, reward, done, info = env.step(action)  # take a random action
        if done:
            print(f"Episode finished. {ts=}")
            break


def main():
    if args.control:
        model, _ = setup_control()
        model.learn(total_timesteps=args.N_iter * args.N_mu * args.N_traj)
        model.save(f'{args.pickle}_control')
    else:
        prot, adv, env = setup_adv()
        if not args.demo_mode:
            train(prot, adv)
        run(env)
        env.close()


if __name__ == '__main__':
    main()
