import abc


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


def get_link_by_name(client, body_i, link_name):
    """
    Get link index by name from body index body_i
    Adapted from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12728
    """
    for link_i in range(client.getNumJoints(body_i)):
        if client.getJointInfo(body_i, link_i)[12].decode('utf-8') == link_name:
            return link_i
    return None


def scale_physics(p, body_i, link_i, mass_coefficient, friction_coefficient):
    """
    Scale mass and friction values of a link
    """
    dynamics = p.getDynamicsInfo(body_i, link_i)
    mass = dynamics[0]
    friction = dynamics[1]
    new_mass = mass * mass_coefficient
    new_friction = friction * friction_coefficient
    p.changeDynamics(body_i, link_i, mass=new_mass,
                     lateralFriction=new_friction)
    changed_dynamics = p.getDynamicsInfo(body_i, link_i)
    assert changed_dynamics[0] == new_mass
    assert changed_dynamics[1] == new_friction
