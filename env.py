import gym
from gym.envs.classic_control.acrobot import *


class BaseAcrobotEnv(AcrobotEnv):
    """Base env for Acrobot task which bridges between main and adversarial agent"""

    def __init__(self):
        super(BaseAcrobotEnv, self).__init__()
        self.adv_agent = None
        self.main_agent = None

    def step_two_actions(self, main_action, adv_action):
        """Step the environment with an action from both the main agent and the adversarial agent."""

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

    def get_ob(self):
        return self._get_ob()

    def link_agents(self, main_agent, adv_agent):
        self.main_agent = main_agent
        self.adv_agent = adv_agent

    def is_linked(self):
        """Returns whether this environment linked with both the main and adversarial agent"""
        return self.adv_agent is not None and self.main_agent is not None


class MainAgentEnv(gym.Env):
    """
    An environment for the main agent to act against adversarial actions.
    Required fields from Stable Baselines v3 (https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, base: BaseAcrobotEnv):
        super(MainAgentEnv, self).__init__()
        self.base = base
        self.action_space = self.base.action_space
        self.observation_space = self.base.observation_space

    def step(self, main_action):
        assert self.base.is_linked()

        prestep_obs = self.base.get_ob()
        adv_action, _ = self.base.adv_agent.predict(prestep_obs)
        poststep_obs, r, d, i = self.base.step_two_actions(main_action, adv_action)
        return poststep_obs, r, d, i

    def reset(self):
        return self.base.reset()

    def render(self, mode='human'):
        return self.base.render(mode)

    def close(self):
        return self.base.close()


class AdversarialAgentEnv(gym.Env):
    """
    An environment for the adversarial agent to act, while the main agent takes actions.
    Required fields from Stable Baselines v3 (https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, base: BaseAcrobotEnv):
        super(AdversarialAgentEnv, self).__init__()
        self.base = base
        self.action_space = self.base.action_space
        self.observation_space = self.base.observation_space

    def step(self, adv_action):
        assert self.base.is_linked()

        prestep_obs = self.base.get_ob()
        main_action, _ = self.base.main_agent.predict(prestep_obs)
        poststep_obs, r, d, i = self.base.step_two_actions(main_action, adv_action)
        return poststep_obs, -r, d, i

    def reset(self):
        return self.base.reset()

    def render(self, mode='human'):
        return self.base.render(mode)

    def close(self):
        return self.base.close()


if __name__ == '__main__':
    from stable_baselines3 import PPO

    # Set up environments
    base_env = BaseAcrobotEnv()
    main_env = MainAgentEnv(base_env)
    adv_env = AdversarialAgentEnv(base_env)

    # Set up agents
    main_agent = PPO("MlpPolicy", main_env, verbose=1)
    adv_agent = PPO("MlpPolicy", adv_env, verbose=1)

    # Link agents
    base_env.link_agents(main_agent, adv_agent)

    # Main agent tries to act
    obs = main_env.reset()
    a, _ = main_agent.predict(obs)
    main_env.step(a)

    # Adversarial agent tries to act
    obs = adv_env.reset()
    a, _ = adv_agent.predict(obs)
    adv_env.step(a)
