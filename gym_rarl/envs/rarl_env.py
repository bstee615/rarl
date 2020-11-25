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

    def reset(self):
        return self.base.reset()

    def render(self, mode='human'):
        return self.base.render(mode)

    def close(self):
        return self.base.close()


class MainRarlEnv(BaseRarlEnv):
    """
    An environment for the main agent to act against adversarial actions.
    """

    def step(self, main_action):
        assert self.bridge.is_linked()

        prestep_obs = self.base.get_ob()
        adv_action, _ = self.bridge.adv_agent.predict(prestep_obs)
        poststep_obs, r, d, i = self.base.step_two_agents(main_action, adv_action)
        return poststep_obs, r, d, i


class AdversarialRarlEnv(BaseRarlEnv):
    """
    An environment for the adversarial agent to act, while the main agent takes actions.
    """

    def step(self, adv_action):
        assert self.bridge.is_linked()

        prestep_obs = self.base.get_ob()
        main_action, _ = self.bridge.main_agent.predict(prestep_obs)
        poststep_obs, r, d, i = self.base.step_two_agents(main_action, adv_action)
        return poststep_obs, -r, d, i


if __name__ == '__main__':
    from stable_baselines3 import PPO
    from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv

    # Set up environments
    base_env = AdversarialCartPoleEnv()

    bridge = Bridge()
    main_env = MainRarlEnv(base_env, bridge)
    adv_env = AdversarialRarlEnv(base_env, bridge)

    # Set up agents
    main_agent = PPO("MlpPolicy", main_env, verbose=1)
    adv_agent = PPO("MlpPolicy", adv_env, verbose=1)

    # Link agents
    bridge.link_agents(main_agent, adv_agent)

    # Main agent tries to act
    obs = main_env.reset()
    a, _ = main_agent.predict(obs)
    main_env.step(a)

    # Adversarial agent tries to act
    obs = adv_env.reset()
    a, _ = adv_agent.predict(obs)
    adv_env.step(a)
