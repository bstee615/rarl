import abc

import gym

from bridge import Bridge
from gym_rarl.envs.adv_env import BaseAdversarialEnv


class BaseRarlEnv(abc.ABC, gym.Env):
    """
    A Gym environment where an agent can act against an adversary.
    Defines required fields from Stable Baselines v3 (https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)
    """

    def __init__(self, base: BaseAdversarialEnv, bridge: Bridge):
        super().__init__()
        self.base = base
        self.bridge = bridge
        self.action_space = self.base.action_space
        self.observation_space = self.base.observation_space

    @abc.abstractmethod
    def step(self, action):
        pass

    def seed(self, seed=None):
        self.base.seed(seed)

    def reset(self):
        return self.base.reset()

    def render(self, mode='human'):
        return self.base.render(mode)

    def close(self):
        return self.base.close()


class ProtagonistRarlEnv(BaseRarlEnv):
    """
    An environment for the main agent to act against adversarial actions.
    """

    def step(self, prot_action):
        prestep_obs = self.base.get_ob()
        # assert self.bridge.is_linked()
        if self.bridge.adv_agent:
            adv_action, _ = self.bridge.adv_agent.predict(prestep_obs)
        else:
            adv_action = None
        # print(prot_action, adv_action)
        poststep_obs, r, d, i = self.base.step_two_agents(prot_action, adv_action)
        return poststep_obs, r, d, i


class AdversarialRarlEnv(BaseRarlEnv):
    """
    An environment for the adversarial agent to act, while the main agent takes actions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = self.adv_action_space

    def step(self, adv_action):
        prestep_obs = self.base.get_ob()
        if self.bridge.prot_agent:
            prot_action, _ = self.bridge.prot_agent.predict(prestep_obs)
        else:
            prot_action = None
        poststep_obs, r, d, i = self.base.step_two_agents(prot_action, adv_action)
        return poststep_obs, -r, d, i


if __name__ == '__main__':
    from stable_baselines3 import PPO
    from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv

    # Set up environments
    base_env = AdversarialCartPoleEnv()

    bridge = Bridge()
    prot_env = ProtagonistRarlEnv(base_env, bridge)
    adv_env = AdversarialRarlEnv(base_env, bridge)

    # Set up agents
    prot_agent = PPO("MlpPolicy", prot_env, verbose=1)
    adv_agent = PPO("MlpPolicy", adv_env, verbose=1)

    # Link agents
    bridge.link_agents(prot_agent, adv_agent)

    # Main agent tries to act
    obs = prot_env.reset()
    a, _ = prot_agent.predict(obs)
    prot_env.step(a)

    # Adversarial agent tries to act
    obs = adv_env.reset()
    a, _ = adv_agent.predict(obs)
    adv_env.step(a)
