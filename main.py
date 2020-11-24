import gym

env = gym.make('CartPole-v0')
for episode in range(20):
    obs = env.reset()
    for ts in range(100):
        env.render()
        print(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)  # take a random action
        if done:
            print(f"Episode finished. {ts=}")
            break
env.close()
