import abc


class BaseAdversarialEnv(abc.ABC):
    """
    Wraps a Gym environment where two actors act in each step.
    """

    @abc.abstractmethod
    def get_ob(self):
        """
        Return the latest observation s_t
        """
        pass

    @abc.abstractmethod
    def step_two_agents(self, prot_action, adv_action):
        """
        Step the environment with an action from each agent.
        """
        pass
