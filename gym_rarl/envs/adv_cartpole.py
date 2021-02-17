from time import sleep

import gym
from gym.envs.classic_control.acrobot import *
from pybullet_envs.bullet import CartPoleBulletEnv

from gym_rarl.envs.adv_env import BaseAdversarialEnv, scale_physics, get_link_by_name


class AdversarialCartPoleBulletEnv(BaseAdversarialEnv, CartPoleBulletEnv):
    """
    Wraps CartPole env and allows two actors to act in each step.
    """

    def __init__(self, mass_percentage=1.0, friction_percentage=1.0, adv_percentage=1.0, **kwargs):
        self.adv_force_mag = 2.0 * adv_percentage  # TODO tune this parameter

        super().__init__(**kwargs)

        self.mass_percentage = mass_percentage
        self.friction_percentage = friction_percentage

    @property
    def adv_action_space(self):
        action_dim = 2
        if self._discrete_actions:
            # 2 discrete actions per dimension
            return spaces.MultiDiscrete([action_dim] * 2)
        else:
            action_high = np.array([self.adv_force_mag] * action_dim)
            return spaces.Box(-action_high, action_high)

    def reset(self):
        obs = super().reset()
        p = self._p
        self.pole_link_i = get_link_by_name(p, self.cartpole, 'pole')
        for link_i in range(p.getNumJoints(self.cartpole)):
            scale_physics(p, self.cartpole, link_i, self.mass_percentage, self.friction_percentage)
        return obs

    def get_ob(self):
        return np.array(self.state)

    def step_two_agents(self, action, adv_action):
        """
        Copied from pybullet_envs.bullet:CartPoleBulletEnv (gym==0.17.3, pybullet==3.0.7)
        """
        p = self._p
        if action is not None:
            if self._discrete_actions:
                force = self.force_mag if action == 1 else -self.force_mag
            else:
                force = action[0]
            p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force)

        if adv_action is not None:
            if self._discrete_actions:
                adv_force_scaled = adv_action * self.adv_force_mag
                adv_force = (adv_force_scaled[0], 0.0, adv_force_scaled[1])
            else:
                adv_force = (adv_action[0], 0.0, adv_action[1])
            p.applyExternalForce(self.cartpole, self.pole_link_i, forceObj=adv_force,
                                 posObj=(0.0, 0.0, 0.0),
                                 flags=p.WORLD_FRAME)

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
    env = gym.make('AdversarialCartPoleBulletEnv-v0', agent='adversarial', renders=True)
    # env = gym.make('CartPoleBulletEnv-v1', renders=True)
    env.reset()
    for _ in range(1000):
        sleep(1 / 30)
        env.step_two_agents(None, env.adv_action_space.sample())
        env.render()
        # env.step_two_agents(env.action_space.sample(), None)
        # env.step(0)
    env.close()


if __name__ == '__main__':
    main()
