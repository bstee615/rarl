from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv, HalfCheetahBulletEnv, \
    HopperBulletEnv, AntBulletEnv

from gym_rarl.envs.base_adv_walker import BaseAdversarialWalkerEnv


class AdversarialWalker2DEnv(BaseAdversarialWalkerEnv, Walker2DBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['foot', 'foot_left']

    def __init__(self, adv_percentage=1.0, **kwargs):
        Walker2DBulletEnv.__init__(self, **kwargs)

        self.adv_force_mag = 250.0 * adv_percentage  # TODO tune this parameter


class AdversarialHalfCheetahEnv(BaseAdversarialWalkerEnv, HalfCheetahBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['torso', 'bfoot', 'ffoot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        HalfCheetahBulletEnv.__init__(self, **kwargs)

        self.adv_force_mag = 250.0 * adv_percentage  # TODO tune this parameter


class AdversarialHopperEnv(BaseAdversarialWalkerEnv, HopperBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['foot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        HopperBulletEnv.__init__(self, **kwargs)

        self.adv_force_mag = 75.0 * adv_percentage  # TODO tune this parameter


class AdversarialAntEnv(BaseAdversarialWalkerEnv, AntBulletEnv):
    @property
    def parts_to_perturb(self):
        return ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, adv_percentage=1.0, **kwargs):
        AntBulletEnv.__init__(self, **kwargs)

        self.adv_force_mag = 600.0 * adv_percentage  # TODO tune this parameter
