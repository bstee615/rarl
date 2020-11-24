import gym
import stable_baselines3
from gym.envs.classic_control.acrobot import *


class BaseEnv(AcrobotEnv):
    def __init__(self, main_agent, adv_agent, friction, blahblah):
        super(BaseEnv, self).__init__()
        self.blahblah = blahblah
        self.friction = friction
        self.adv_agent = adv_agent
        self.main_agent = main_agent

    def special_step(self, main_action, adv_action):
        # s = self.state
        # torque = self.AVAIL_TORQUE[main_action]+adv_action
        #
        # # Add noise to the force action
        # if self.torque_noise_max > 0:
        #     torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)
        #
        # # Now, augment the state with our force action so it can be passed to
        # # _dsdt
        # s_augmented = np.append(s, torque)
        #
        # ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # # only care about final timestep of integration returned by integrator
        # ns = ns[-1]
        # ns = ns[:4]  # omit action
        # # ODEINT IS TOO SLOW!
        # # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # # self.s_continuous = ns_continuous[-1] # We only care about the state
        # # at the ''final timestep'', self.dt
        #
        # ns[0] = wrap(ns[0], -pi, pi)
        # ns[1] = wrap(ns[1], -pi, pi)
        # ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        # ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        # self.state = ns
        # terminal = self._terminal()
        # reward = -1. if not terminal else 0.
        # return (self._get_ob(), reward, terminal, {})
        self.AVAIL_TORQUE = adv_action
        return self.step(main_action)

    def get_ob(self):
        return self._get_ob()


class MainEnv(gym.Env):
    def __init__(self, base: BaseEnv):
        super(MainEnv, self).__init__()
        self.base = base

    def step(self, main_action):
        adv = self.adv_agent
        # TODO test this line
        adv_action = self.adv_agent.predict(self.base.get_ob())
        o, r, d, i = self.base.speical_step(main_action, adv_action)
        return o, r, d, i

    def set_adv_action(self, adv_action):
        self.adv_action = adv_action


class AdvEnv(gym.Env):
    def __init__(self, base):
        super(AdvEnv, self).__init__()
        self.base = base

    def step(self, adv_action):
        main = self.base.main_agent
        main_action = main.get_action()
        o, r, d, i = self.base.special_step(main_action, adv_action)
        return o, r, d, i
