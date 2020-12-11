from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines3 import PPO
# Set up environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import MainRarlEnv, AdversarialRarlEnv

N_steps = 64

# %% Train the agent
N_iter = 10
N_mu = 2
N_nu = 2
N_traj = 128
N_traj_over_n_steps = N_traj / N_steps
assert N_steps % 2 == 0
assert int(N_traj_over_n_steps)  # works


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
    for i in range(N_iter):
        for j in range(N_mu):
            # rollout N_traj timesteps training the protagonist
            prot.learn(total_timesteps=N_traj_over_n_steps, reset_num_timesteps=False)
        for j in range(N_nu):
            # rollout N_traj timesteps training the adversary
            adv.learn(total_timesteps=N_traj_over_n_steps, reset_num_timesteps=False)
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
