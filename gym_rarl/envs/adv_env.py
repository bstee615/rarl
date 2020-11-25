import abc

import gym


class BaseAdversarialEnvWrapper(abc.ABC, gym.core.Env):
    """
    Base environment wrapping a Gym task where both a main agent and adversary act in each step.
    """

    def __init__(self):
        super().__init__()
        self.adv_agent = None
        self.main_agent = None

    def link_agents(self, main_agent, adv_agent):
        self.main_agent = main_agent
        self.adv_agent = adv_agent

    def is_linked(self):
        """
        Returns whether this environment linked with both the main and adversarial agent
        """
        return self.adv_agent is not None and self.main_agent is not None

    @abc.abstractmethod
    def get_ob(self):
        """
        Return the latest observation s_t
        """
        pass

    @abc.abstractmethod
    def step_two_actors(self, main_action, adv_action):
        """
        Step the environment with an action from both the main agent and the adversarial agent.
        """
        pass
