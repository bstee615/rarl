from time import sleep

from gym.envs.classic_control.acrobot import *
from pybullet_envs.bullet import CartPoleBulletEnv

from gym_rarl.envs.adv_env import BaseAdversarialEnv


class AdversarialCartPoleEnv(BaseAdversarialEnv, CartPoleBulletEnv):
    """
    Wraps CartPole env and allows two actors to act in each step.
    """

    def __init__(self, *args, **kwargs):
        self.apply_adv = kwargs.pop('apply_adv', True)
        self.apply_prot = kwargs.pop('apply_prot', True)
        CartPoleBulletEnv.__init__(self, *args, **kwargs)

        self.adv_force_mag = self.force_mag

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

        if self.apply_adv:
            p.applyExternalForce(self.cartpole, 1, forceObj=(adv_force, 0, 0), posObj=(0, 0, 0), flags=p.WORLD_FRAME)

        if self.apply_prot:
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
    env = AdversarialCartPoleEnv(renders=True)
    env.reset()
    for _ in range(1000):
        env.render()
        sleep(1 / 30)
        env.step_two_agents(0, 1)
        # env.step(0)
    env.close()


if __name__ == '__main__':
    main()
