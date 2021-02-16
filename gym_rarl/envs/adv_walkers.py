from time import sleep

import gym
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv, HalfCheetahBulletEnv, \
    HopperBulletEnv, AntBulletEnv

from gym_rarl.envs.base_adv_walker import BaseAdversarialWalkerEnv


class AdversarialWalker2DBulletEnv(BaseAdversarialWalkerEnv, Walker2DBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['foot', 'foot_left']

    def __init__(self, adv_percentage=1.0, **kwargs):
        super().__init__(**kwargs)

        self.adv_force_mag = 50.0 * adv_percentage  # TODO tune this parameter


class AdversarialHalfCheetahBulletEnv(BaseAdversarialWalkerEnv, HalfCheetahBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['torso', 'bfoot', 'ffoot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        super().__init__(**kwargs)

        self.adv_force_mag = 62.5 * adv_percentage  # TODO tune this parameter


class AdversarialHopperBulletEnv(BaseAdversarialWalkerEnv, HopperBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['foot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        super().__init__(**kwargs)

        self.adv_force_mag = 18.75 * adv_percentage  # TODO tune this parameter


class AdversarialAntBulletEnv(BaseAdversarialWalkerEnv, AntBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        super().__init__(**kwargs)

        self.adv_force_mag = 500.0 * adv_percentage  # TODO tune this parameter


def main():
    env = gym.make('AdversarialHopperBulletEnv-v0', render=True)
    env.reset()
    for _ in range(1000):
        sleep(1 / 30)
        env.step_two_agents(env.action_space.sample(), env.adv_action_space.sample())
        # env.step_two_agents(None, np.array([1.0] * 2))
        # env.step_two_agents(None, None)
        env.render()
        # env.step_two_agents(env.action_space.sample(), None)
        # env.step(0)
    env.close()


if __name__ == '__main__':
    main()
