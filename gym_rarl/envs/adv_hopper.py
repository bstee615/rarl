from time import sleep

from pybullet_envs.gym_locomotion_envs import HopperBulletEnv

from gym_rarl.envs.adv_env import get_link_by_name
from gym_rarl.envs.base_adv_walker import BaseAdversarialWalkerEnv


class AdversarialHopperEnv(BaseAdversarialWalkerEnv, HopperBulletEnv):
    """
    Wraps Bullet env and allows two actors to act in each step.
    """

    @property
    def adv_action_space_dim(self):
        return 2

    def __init__(self, adv_percentage=1.0, renders=False, **kwargs):
        HopperBulletEnv.__init__(self, render=renders, **kwargs)

        self.adv_force_mag = 75.0 * adv_percentage  # TODO tune this parameter

    def apply_adv_action(self, adv_action):
        p = self.robot._p
        body_i = self.robot.robot_body.bodies[0]
        foot_link_i = get_link_by_name(p, body_i, 'foot')
        p.applyExternalForce(
            body_i, foot_link_i,
            forceObj=(
                adv_action[0] * self.adv_force_mag,
                0.0,  # y = 0
                adv_action[1] * self.adv_force_mag,
            ),
            posObj=(0.0, 0.0, 0.0),
            flags=p.WORLD_FRAME)


def main():
    env = AdversarialHopperEnv(renders=True, adv_percentage=1.0)
    env.reset()
    for _ in range(1000):
        env.render()
        sleep(1 / 30)
        env.step_two_agents(None, [-1.0, 1.0])
        # env.step(0)
    env.close()


if __name__ == '__main__':
    main()
