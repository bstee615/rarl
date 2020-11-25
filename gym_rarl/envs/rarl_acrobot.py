import abc

import gym
from gym.envs.classic_control.acrobot import *


class BaseAdversarialEnvWrapper(abc.ABC, gym.core.Env):
    """
    Base environment wrapping a Gym task where both a main agent and adversary act in each step
    """

    def __init__(self):
        super().__init__()
        self.adv_agent = None
        self.main_agent = None

    def link_agents(self, main_agent, adv_agent):
        self.main_agent = main_agent
        self.adv_agent = adv_agent

    def is_linked(self):
        """Returns whether this environment linked with both the main and adversarial agent"""
        return self.adv_agent is not None and self.main_agent is not None

    @abc.abstractmethod
    def get_ob(self):
        """
        Return the latest observation s_t
        """
        pass

    @abc.abstractmethod
    def step_two_actions(self, main_action, adv_action):
        """
        Step the environment with an action from both the main agent and the adversarial agent.
        """
        pass


class AdversarialAcrobotEnvWrapper(BaseAdversarialEnvWrapper, AcrobotEnv):
    """Base env for Acrobot task which bridges between main and adversarial agent"""

    def get_ob(self):
        return self._get_ob()

    def step_two_actions(self, main_action, adv_action):
        """
        Copied from AcrobotEnv (gym==0.17.3)
        """

        s = self.state
        torque = self.AVAIL_TORQUE[main_action]  # TODO add torque from adversarial action

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})
