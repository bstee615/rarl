from time import sleep

from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv

from gym_rarl.envs.base_adv_walker import BaseAdversarialWalkerEnv


class AdversarialHalfCheetahEnv(BaseAdversarialWalkerEnv, HalfCheetahBulletEnv):
    """
    Wraps Bullet env and allows two actors to act in each step.
    """

    @property
    def parts_to_perturb(self):
        return ['torso', 'bfoot', 'ffoot']

    def __init__(self, adv_percentage=1.0, renders=False, **kwargs):
        HalfCheetahBulletEnv.__init__(self, render=renders, **kwargs)

        self.adv_force_mag = 250.0 * adv_percentage  # TODO tune this parameter


def main():
    env = AdversarialHalfCheetahEnv(renders=True, adv_percentage=1.0)
    env.reset()
    for _ in range(1000):
        env.render()
        sleep(1 / 30)
        env.step_two_agents(None, env.adv_action_space.sample())
        # env.step(0)
    env.close()


if __name__ == '__main__':
    main()
