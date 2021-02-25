import gym
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv, HalfCheetahBulletEnv, \
    HopperBulletEnv, AntBulletEnv

from gym_rarl.envs.base_adv_walker import BaseAdversarialWalkerEnv


class AdversarialWalker2DBulletEnv(BaseAdversarialWalkerEnv, Walker2DBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['foot', 'foot_left']

    def __init__(self, adv_force=None, **kwargs):
        super().__init__(**kwargs)

        if adv_force is None:
            self.adv_force_mag = 1.5625  # Tuned after 3 trains and normalized
        else:
            self.adv_force_mag = adv_force


class AdversarialHalfCheetahBulletEnv(BaseAdversarialWalkerEnv, HalfCheetahBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['torso', 'bfoot', 'ffoot']

    def __init__(self, adv_force=None, **kwargs):
        super().__init__(**kwargs)

        if adv_force is None:
            self.adv_force_mag = 0.5  # Tuned after 3 trains and normalized
        else:
            self.adv_force_mag = adv_force


class AdversarialHopperBulletEnv(BaseAdversarialWalkerEnv, HopperBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['foot']

    def __init__(self, adv_force=None, **kwargs):
        super().__init__(**kwargs)

        if adv_force is None:
            self.adv_force_mag = 1.1875  # Tuned after 3 trains and normalized
        else:
            self.adv_force_mag = adv_force


class AdversarialAntBulletEnv(BaseAdversarialWalkerEnv, AntBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, adv_force=None, **kwargs):
        super().__init__(**kwargs)

        if adv_force is None:
            self.adv_force_mag = 5.625  # Tuned after 4 trains and normalized
        else:
            self.adv_force_mag = adv_force


def main():
    env = gym.make('AdversarialAntBulletEnv-v0', render=True)
    env.reset()
    for _ in range(1000):
        # sleep(1 / 60)
        env.step_two_agents(env.action_space.sample(), env.adv_action_space.sample())
        # env.step_two_agents(None, np.array([1.0] * 2))
        # env.step_two_agents(None, None)
        env.render()
        # env.step_two_agents(env.action_space.sample(), None)
        # env.step(0)
    env.close()


if __name__ == '__main__':
    main()
