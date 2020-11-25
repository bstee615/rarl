import abc

import gym


class BaseBridgeEnv(abc.ABC, gym.core.Env):
    """
    Base environment wrapping a Gym task where two actors act in each step.
    """

    def __init__(self):
        super().__init__()
        self.adv_agent = None
        self.main_agent = None

    def link_agents(self, main_agent, adv_agent):
        """
        Link main_agent and adv_agent. These are the two agents which will be taking actions.
        """
        self.main_agent = main_agent
        self.adv_agent = adv_agent

    def is_linked(self):
        """
        Returns whether this environment is linked to both the main and adversarial agent
        """
        return self.adv_agent is not None and self.main_agent is not None

    @abc.abstractmethod
    def get_ob(self):
        """
        Return the latest observation s_t
        """
        pass

    @abc.abstractmethod
    def step_two_agents(self, main_action, adv_action):
        """
        Step the environment with an action from each agents.
        """
        pass
