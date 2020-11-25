from stable_baselines3 import PPO

# Set up environments
from gym_rarl.envs.adv_acrobot import AdversarialAcrobotEnvWrapper
from gym_rarl.envs.rarl_env import MainRarlEnv, AdversarialRarlEnv

base_env = AdversarialAcrobotEnvWrapper()
main_env = MainRarlEnv(base_env)
adv_env = AdversarialRarlEnv(base_env)

# Set up agents
main_agent = PPO("MlpPolicy", main_env, verbose=1)
adv_agent = PPO("MlpPolicy", adv_env, verbose=1)

# Link agents
base_env.link_agents(main_agent, adv_agent)

# Main agent tries to act
obs = main_env.reset()
a, _ = main_agent.predict(obs)
main_env.step(a)

# Adversarial agent tries to act
obs = adv_env.reset()
a, _ = adv_agent.predict(obs)
adv_env.step(a)

env = main_env
model = main_agent

# %% Train the agent
model.learn(total_timesteps=10000)
model.save("models/ppo-acrobot-bleh")

# %%
model.load("models/ppo-acrobot-bleh")
obs = env.reset()
for ts in range(5000):
    env.render()
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)  # take a random action
    if done:
        print(f"Episode finished. {ts=}")
        break
env.close()
