import abc


def get_link_by_name(client, body_i, link_name):
    """
    Get link index by name from body index body_i
    Adapted from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12728
    """
    for link_i in range(client.getNumJoints(body_i)):
        if client.getJointInfo(body_i, link_i)[12].decode('utf-8') == link_name:
            return link_i
    return None


class BaseAdversarialEnv(abc.ABC):
    """
    Wraps a Gym environment where two actors act in each step.
    """

    @property
    @abc.abstractmethod
    def adv_action_space(self):
        pass

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
