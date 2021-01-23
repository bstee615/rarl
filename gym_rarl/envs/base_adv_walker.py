import gym
import numpy as np
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv

from gym_rarl.envs.adv_env import BaseAdversarialEnv, get_link_by_name


class BaseAdversarialWalkerEnv(BaseAdversarialEnv, WalkerBaseBulletEnv):
    """
    Base class for environments with walker base (hopper, ant, etc).
    """

    @property
    def parts_to_perturb(self):
        raise NotImplementedError()

    def get_ob(self):
        return self.robot.calc_state()

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

        return state, sum(self.rewards), bool(done), {}

    def apply_adv_action(self, adv_action):
        p = self.robot._p
        body_i = self.robot.robot_body.bodies[0]
        for i, name in enumerate(self.parts_to_perturb):
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
