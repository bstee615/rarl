import time

from stable_baselines3 import PPO

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import MainRarlEnv, AdversarialRarlEnv

bridge = Bridge()
base_env = AdversarialCartPoleEnv(renders=True, apply_adv=True, apply_prot=False)
prot_env = MainRarlEnv(base_env, bridge)
adv_env = AdversarialRarlEnv(base_env, bridge)

# Set up agents
prot = PPO("MlpPolicy", prot_env)
adv = PPO("MlpPolicy", adv_env)

# Link agents
bridge.link_agents(prot, adv)


# prot, adv, prot_env, adv_env = main.setup_adv()

def do(agent, env, func):
    obs = env.reset()
    directions = [0, 0]
    for i in range(100):
        action = func(agent, obs)[0]
        obs, reward, done, _ = env.step(action)
        time.sleep(1 / 60)
        # print(i, action, reward)
        directions[action] += 1
        env.render()

    return directions


def odds(l):
    smaller = min(l)
    if smaller == 0:
        lodds = l
    else:
        lodds = [d / smaller for d in directions]
    return lodds


for func in [
    lambda a, obs: a.predict(obs),
    lambda a, obs: (1, None),
    lambda a, obs: (0, None)
]:
    directions = do(adv, adv_env, func)
    directions_odds = odds(directions)
    print('done. Distribution:', directions, directions_odds)

prot_env.close()
adv_env.close()
