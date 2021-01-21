from time import sleep

from gym.envs.classic_control.acrobot import *
from pybullet_envs.bullet import CartPoleBulletEnv

from gym_rarl.envs.adv_env import BaseAdversarialEnv, get_link_by_name


class AdversarialCartPoleEnv(BaseAdversarialEnv, CartPoleBulletEnv):
    """
    Wraps CartPole env and allows two actors to act in each step.
    """

    @property
    def adv_action_space(self):
        return self.action_space

    def __init__(self, adv_percentage, **kwargs):
        CartPoleBulletEnv.__init__(self, **kwargs)

        self.adv_force_mag = 0.05 * adv_percentage  # TODO tune this parameter

    def get_ob(self):
        return np.array(self.state)

    def step_two_agents(self, action, adv_action):
        """
        Copied from pybullet_envs.bullet:CartPoleBulletEnv (gym==0.17.3, pybullet==3.0.7)
        TODO: Currently we consider only discrete actions, the default (self._discrete_actions==True)
        """
        p = self._p
        if self._discrete_actions:
            force = self.force_mag if action == 1 else -self.force_mag
            adv_force = self.adv_force_mag if adv_action == 1 else -self.adv_force_mag
        else:
            raise Exception('benjis: continuous action space not supported')
            force = action[0]

        if adv_action is not None:
            pole_link_i = get_link_by_name(p, self.cartpole, 'pole')
            p.applyExternalForce(self.cartpole, pole_link_i, forceObj=(adv_force, 0.0, 0.0), posObj=(0.0, 0.0, 0.0),
                                 flags=p.LINK_FRAME)

        if action is not None:
            p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force)

        p.stepSimulation()

        self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot, x, x_dot = self.state

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)
        reward = 1.0

        return np.array(self.state), reward, done, {}


def main():
    env = AdversarialCartPoleEnv(renders=True, adv_percentage=1.0)
    env.reset()
    for _ in range(1000):
        env.render()
        sleep(1 / 30)
        env.step_two_agents(0, 1)
        # env.step(0)
    env.close()


if __name__ == '__main__':
    main()
