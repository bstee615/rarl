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

        self.adv_force_mag = 12.5 * adv_percentage  # Tuned after 2 trains


class AdversarialHalfCheetahBulletEnv(BaseAdversarialWalkerEnv, HalfCheetahBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['torso', 'bfoot', 'ffoot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        super().__init__(**kwargs)

        self.adv_force_mag = 10.0 * adv_percentage  # Tuned after 2 trains


class AdversarialHopperBulletEnv(BaseAdversarialWalkerEnv, HopperBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['foot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        super().__init__(**kwargs)

        self.adv_force_mag = 18.75 * adv_percentage  # Tuned after 1 train


class AdversarialAntBulletEnv(BaseAdversarialWalkerEnv, AntBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        super().__init__(**kwargs)

        self.adv_force_mag = 125.0 * adv_percentage  # Tuned after 2 trains


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
