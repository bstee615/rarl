import time

from stable_baselines3 import PPO

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.rarl_env import ProtagonistRarlEnv
from main import dummy

bridge = Bridge()
base_env = AdversarialCartPoleEnv(renders=True)
prot_env = dummy(lambda: ProtagonistRarlEnv(base_env, bridge, apply_adv=False),
                 evaluate_name='models/colab-example_controlenv')
# adv_env = dummy(lambda: AdversarialRarlEnv(base_env, bridge), evaluate_name='control')

# Set up agents
prot = PPO.load('models/colab-example_control.zip', prot_env)
# adv = PPO.load('models/eval-1_control.zip', adv_env)
# prot = PPO("MlpPolicy", prot_env)
# adv = PPO("MlpPolicy", adv_env)

# Link agents
bridge.link_agents(prot, None)


# prot, adv, prot_env, adv_env = main.setup_adv()

def do(agent, env, func):
    obs = env.reset()
    directions = [0, 0]
    for i in range(10000):
        action = func(agent, obs)[0]
        obs, reward, done, _ = env.step(action)
        time.sleep(1 / 45)
        # print(i, action, reward)
        directions[action[0]] += 1
        env.render()
        if done:
            break

    return directions


def odds(l):
    smaller = min(l)
    if smaller == 0:
        lodds = l
    else:
        lodds = [d / smaller for d in directions]
    return lodds


for i in range(10):
    directions = do(prot, prot_env, lambda a, obs: a.predict(obs))
    directions_odds = odds(directions)
    print('done. Distribution:', directions, directions_odds)

prot_env.close()
# adv_env.close()
