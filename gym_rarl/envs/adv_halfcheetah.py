from time import sleep

from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv

from gym_rarl.envs.adv_env import get_link_by_name
from gym_rarl.envs.base_adv_walker import BaseAdversarialWalkerEnv


class AdversarialHalfCheetahEnv(BaseAdversarialWalkerEnv, HalfCheetahBulletEnv):
    """
    Wraps Bullet env and allows two actors to act in each step.
    """

    @property
    def adv_action_space_dim(self):
        return 6

    def __init__(self, adv_percentage=1.0, renders=False, **kwargs):
        HalfCheetahBulletEnv.__init__(self, render=renders, **kwargs)

        self.adv_force_mag = 250.0 * adv_percentage  # TODO tune this parameter

    def apply_adv_action(self, adv_action):
        p = self.robot._p
        body_i = self.robot.robot_body.bodies[0]
        for i, name in enumerate(['torso', 'bfoot', 'ffoot']):
            link_i = get_link_by_name(p, body_i, name)
            action_i = i * 2
            p.applyExternalForce(
                body_i, link_i,
                forceObj=(
                    adv_action[action_i] * self.adv_force_mag,
                    0.0,  # y = 0
                    adv_action[action_i + 1] * self.adv_force_mag,
                ),
                posObj=(0.0, 0.0, 0.0),
                flags=p.WORLD_FRAME)


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
