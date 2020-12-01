import math

from gym.envs.classic_control import CartPoleEnv
from gym.envs.classic_control.acrobot import *

from gym_rarl.envs.adv_env import BaseAdversarialEnv


class AdversarialCartPoleEnv(BaseAdversarialEnv, CartPoleEnv):
    """
    Wraps CartPole env and allows two actors to act in each step.
    """

    def get_ob(self):
        return np.array(self.state)

    def step_two_agents(self, main_action, adv_action):
        """
        Copied from gym.envs.classic_control.cartpole.CartPoleEnv (gym==0.17.3)
        """
        err_msg = "%r (%s) invalid" % (main_action, type(main_action))
        assert self.action_space.contains(main_action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if main_action == 1 else -self.force_mag  # TODO apply adv_action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self.get_ob(), reward, done, {}
