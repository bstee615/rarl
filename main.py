from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines3 import PPO
# Set up environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import MainRarlEnv, AdversarialRarlEnv


def dummy(env_type, seed):
    env = make_vec_env(env_type, seed=seed)
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=10.)
    return env


def adversarial_setup():
    base_env = AdversarialCartPoleEnv(renders=True)
    base_env.seed(100)

    bridge = Bridge()
    main_env = dummy(lambda: MainRarlEnv(base_env, bridge), seed=100)
    adv_env = dummy(lambda: AdversarialRarlEnv(base_env, bridge), seed=100)
    main_env.seed(100)
    adv_env.seed(100)

    # Set up agents
    main_agent = PPO("MlpPolicy", main_env, verbose=True, seed=123456)
    adv_agent = PPO("MlpPolicy", adv_env, verbose=True, seed=123456)

    # Link agents
    bridge.link_agents(main_agent, adv_agent)

    env = main_env
    model = main_agent
    return model, env


def control_setup():
    env = dummy(lambda: CartPoleBulletEnv(renders=True), seed=0)
    env.seed(0)
    model = PPO("MlpPolicy", env, verbose=True, seed=123456)
    return model, env


model, env = adversarial_setup()
# model, env = control_setup()

# %% Train the agent
model.learn(total_timesteps=200000)
model.save("models/ppo-cartpolebullet-200k")
del model

# %%
model = PPO.load("models/ppo-cartpolebullet-200k")
obs = env.reset()
for ts in range(10000):
    env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)  # take a random action
    if done:
        print(f"Episode finished. {ts=}")
        break
env.close()
