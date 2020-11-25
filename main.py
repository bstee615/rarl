import gym
from stable_baselines3 import PPO

env = gym.make('Acrobot-v1')

model = PPO("MlpPolicy", env, verbose=1)
# Train the agent
model.learn(total_timesteps=1000000)
model.save("ppo-acrobot-bleh")

model.load("ppo-acrobot-bleh")
obs = env.reset()
for ts in range(5000):
    env.render()
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)  # take a random action
    if done:
        print(f"Episode finished. {ts=}")
        break
env.close()
