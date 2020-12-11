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
    args = parser.parse_args()

    assert args.N_steps % 2 == 0
    assert isinstance(args.N_traj / args.N_steps, int)

    return args


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
    base_env = AdversarialCartPoleEnv(renders=True)
    base_env.seed(100)

    bridge = Bridge()
    main_env = dummy(lambda: MainRarlEnv(base_env, bridge), seed=100)
    adv_env = dummy(lambda: AdversarialRarlEnv(base_env, bridge), seed=100)
    main_env.seed(100)
    adv_env.seed(100)

    # Set up agents
    prot_agent = PPO("MlpPolicy", main_env, verbose=True, seed=123456)
    adv_agent = PPO("MlpPolicy", adv_env, verbose=False, seed=123456)

    # Link agents
    bridge.link_agents(prot_agent, adv_agent)

    env = main_env
    return prot_agent, adv_agent, env


def setup_control():
    """
    Setup a normal model and environment a a control
    """
    env = dummy(lambda: CartPoleBulletEnv(renders=True), seed=0)
    env.seed(0)
    model = PPO("MlpPolicy", env, verbose=True, seed=123456)
    return model, env


def train(prot, adv):
    """
    Train according to Algorithm 1
    """
    for i in range(args.N_iter):
        for j in range(args.N_mu):
            # rollout N_traj timesteps training the protagonist
            prot.learn(total_timesteps=args.N_traj_over_n_steps, reset_num_timesteps=False)
        for j in range(args.N_nu):
            # rollout N_traj timesteps training the adversary
            adv.learn(total_timesteps=args.N_traj_over_n_steps, reset_num_timesteps=False)
    prot.save("models/ppo-rarl-butt")
    del prot


def demo(env):
    """
    Run model in env
    """
    model = PPO.load("models/ppo-rarl-butt")
    obs = env.reset()
    for ts in range(10000):
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  # take a random action
        if done:
            print(f"Episode finished. {ts=}")
            break


def main():
    prot, adv, env = setup_adv()
    # train(prot, adv)
    demo(env)
    env.close()


if __name__ == '__main__':
    main()
