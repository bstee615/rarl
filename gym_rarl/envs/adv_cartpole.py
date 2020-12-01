from gym.envs.classic_control.acrobot import *
from pybullet_envs.bullet import CartPoleBulletEnv

from gym_rarl.envs.adv_env import BaseAdversarialEnv


class AdversarialCartPoleEnv(BaseAdversarialEnv, CartPoleBulletEnv):
    """
    Wraps CartPole env and allows two actors to act in each step.
    """

    def get_ob(self):
        return np.array(self.state)

    def step_two_agents(self, action, adv_action):
        """
        Copied from pybullet_envs.bullet:CartPoleBulletEnv (gym==0.17.3, pybullet==3.0.7)
        TODO: Currently we consider only discrete actions, the default (self._discrete_actions==True)
        """
        p = self._p
        if self._discrete_actions:
            force = self.force_mag if action == 1 else -self.force_mag
            adv_force = [self.force_mag, 0, 0] if adv_action == 1 else [-self.force_mag, 0,
                                                                        0] if adv_action == -1 else [0, 0, 0]
        else:
            raise Exception('benjis: continuous action space not supported')
            # force = action[0]

        p.applyExternalForce(self.cartpole, 1, forceObj=adv_force, posObj=[0, 0, 0], flags=p.WORLD_FRAME)

        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force)
        p.stepSimulation()

        self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot, x, x_dot = self.state

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)
        reward = 1.0

        return np.array(self.state), reward, done, {}
