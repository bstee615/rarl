import unittest

import numpy as np
import pybullet_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bridge import Bridge
from gym_rarl import envs

verbose = False
render = False
ts = 300
seed = 123
env_pairs = [
    ('HopperBulletEnv-v0', 'AdversarialHopperBulletEnv-v0'),
    # ('CartPoleBulletEnv-v1', 'AdversarialCartPoleBulletEnv-v0'), # TODO fix CartPole
    ('Walker2DBulletEnv-v0', 'AdversarialWalker2DBulletEnv-v0'),
    ('HalfCheetahBulletEnv-v0', 'AdversarialHalfCheetahBulletEnv-v0'),
    ('AntBulletEnv-v0', 'AdversarialAntBulletEnv-v0'),
]


def learn_love_log(agent, env):
    """
    Normalize env with seed, instantiate an agent, and train
    """
    initial_obs = env.reset()
    agent.learn(ts)
    last_obs = env.reset()
    last_prediction, _ = agent.predict(last_obs)
    env.close()
    return initial_obs, last_obs, last_prediction


def init_adv(adv_env_id, disable_adv=False, env_kwargs=None):
    bridge = Bridge()
    default_env_kwargs = {'renders' if 'CartPole' in adv_env_id else 'render': render}
    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs.update(default_env_kwargs)
    env = make_vec_env(adv_env_id, env_kwargs=env_kwargs, seed=seed)
    env = VecNormalize(env)
    prot_agent = PPO('MlpPolicy', env, verbose=verbose, seed=seed, n_steps=ts, bridge=bridge, is_protagonist=True)
    if disable_adv:
        bridge.link_agents(prot_agent, None)
    else:
        adv_agent = PPO('MlpPolicy', env, verbose=verbose, seed=seed, n_steps=ts, bridge=bridge, is_protagonist=False)
        bridge.link_agents(prot_agent, adv_agent)
    return prot_agent, env


def init_control(env_id):
    env_kwargs = {'renders' if 'CartPole' in env_id else 'render': render}
    env = make_vec_env(env_id, env_kwargs=env_kwargs, seed=seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    agent = PPO('MlpPolicy', env, verbose=verbose, seed=seed, n_steps=ts)
    return agent, env


class RarlSanityTests(unittest.TestCase):
    def test_envs_are_registered(self):
        for e, a in env_pairs:
            assert f'- {e}' in pybullet_envs.getList()
            assert a in envs.getList()

    def test_null_adversary_acts_the_same(self):
        for env_id, adv_env_id in env_pairs:
            print(env_id, adv_env_id)
            # do control
            agent, env = init_control(env_id)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary disengaged
            prot_agent, prot_env = init_adv(adv_env_id, disable_adv=True)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert np.array_equal(last_adv_obs, last_control_obs)

    def test_acting_adversary_acts_different(self):
        for env_id, adv_env_id in env_pairs:
            agent, env = init_control(env_id)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary engaged
            prot_agent, prot_env = init_adv(adv_env_id)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert not np.array_equal(last_adv_obs, last_control_obs)

    def test_different_mass_acts_different(self):
        for env_id, adv_env_id in env_pairs:
            agent, env = init_control(env_id)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary engaged
            prot_agent, prot_env = init_adv(adv_env_id, disable_adv=True, env_kwargs={"mass_percentage": 0.5})
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert not np.array_equal(last_adv_obs, last_control_obs)

    def test_different_friction_acts_different(self):
        for env_c, adv_env_c in env_pairs:
            agent, env = init_control(env_c)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary engaged
            prot_agent, prot_env = init_adv(adv_env_c, env_kwargs={"friction_percentage": 0.5}, disable_adv=True)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            if 'CartPole' not in adv_env_c:
                assert not np.array_equal(last_adv_obs, last_control_obs)


if __name__ == '__main__':
    unittest.main()
