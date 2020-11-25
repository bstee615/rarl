from gym.envs.classic_control.acrobot import *

from gym_rarl.envs.adv_env import BaseAdversarialEnvWrapper


class AdversarialAcrobotEnvWrapper(BaseAdversarialEnvWrapper, AcrobotEnv):
    """
    Base env for Acrobot task which allows main/adv to act in each step
    """

    def get_ob(self):
        return self._get_ob()

    def step_two_actors(self, main_action, adv_action):
        """
        Copied from AcrobotEnv (gym==0.17.3)
        """

        s = self.state
        torque = self.AVAIL_TORQUE[main_action]  # TODO add torque from adversarial action

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})
