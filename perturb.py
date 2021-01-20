import time

from stable_baselines3 import PPO

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import AdversarialRarlEnv

bridge = Bridge()
base_env = AdversarialCartPoleEnv(renders=True, adv_percentage=10.0)
# prot_env = ProtagonistRarlEnv(base_env, bridge)
adv_env = AdversarialRarlEnv(base_env, bridge)

# prot = PPO('MlpPolicy', prot_env)
adv = PPO('MlpPolicy', adv_env)

# Link agents
bridge.link_agents(None, adv)


def do(agent, env, func):
    obs = env.reset()
    directions = [0, 0]
    for i in range(10000):
        action = 0
        obs, reward, done, _ = env.step(action)
        time.sleep(1 / 45)
        # directions[action[0]] += 1
        env.render()
        if done:
            # break
            pass
    print(directions)


do(adv, adv_env, lambda a, o: ([1], None))

# prot_env.close()
adv_env.close()
