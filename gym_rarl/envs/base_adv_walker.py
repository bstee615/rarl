import gym
import numpy as np
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv

from gym_rarl.envs.adv_env import BaseAdversarialEnv, get_link_by_name, scale_physics


class BaseAdversarialWalkerEnv(BaseAdversarialEnv, WalkerBaseBulletEnv):
    """
    Base class for environments with walker base (hopper, ant, etc).
    """

    def __init__(self, mass_percentage=1.0, friction_percentage=1.0, simple_reward=False, **kwargs):
        super().__init__(**kwargs)
        self.links = []
        self.mass_percentage = mass_percentage
        self.friction_percentage = friction_percentage
        self.simple_reward = simple_reward

    @property
    def parts_to_perturb(self):
        raise NotImplementedError()

    def get_ob(self):
        return self.robot.calc_state()

    def reset(self):
        obs = super().reset()
        p = self.robot._p
        body_i = self.robot.robot_body.bodies[0]
        self.links = [get_link_by_name(p, body_i, link_name) for link_name in self.parts_to_perturb]
        for link_i in range(p.getNumJoints(body_i)):
            scale_physics(p, body_i, link_i, self.mass_percentage, self.friction_percentage)
        return obs

    @property
    def adv_action_space(self):
        bounds_mag = np.ones([self.adv_action_space_dim])
        return gym.spaces.Box(-bounds_mag, bounds_mag)

    @property
    def adv_action_space_dim(self):
        return len(self.parts_to_perturb) * 2

    def step_two_agents(self, action, adv_action):
        """
        Copied from pybullet (gym==0.17.3, pybullet==3.0.7)
        """
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            if action is not None:
                self.robot.apply_action(action)  # protagonist action
            if adv_action is not None:
                self.apply_adv_action(adv_action)  # antagonist action
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        if action is None:
            electricity_cost = 0
        else:
            electricity_cost = self.electricity_cost * float(np.abs(action * self.robot.joint_speeds).mean(
            ))  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(action).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, action, done)
        self.reward += sum(self.rewards)
        reward = progress if self.simple_reward else sum(self.rewards)
        return state, reward, bool(done), {}

    def apply_adv_action(self, adv_action):
        """
        Apply adversary's action.
        Dimension should respond with the dimension of self.adv_action_space. 
        """
        p = self.robot._p
        body_i = self.robot.robot_body.bodies[0]
        for i, link_i in enumerate(self.links):
            action_i = i * 2
            force_obj = (
                adv_action[action_i] * self.adv_force_mag,
                0.0,  # y = 0
                adv_action[action_i + 1] * self.adv_force_mag,
            )
            p.applyExternalForce(
                body_i, link_i,
                forceObj=force_obj,
                posObj=(0.0, 0.0, 0.0),
                flags=p.WORLD_FRAME)
