import unittest
from functools import partial

import numpy as np
from pybullet_envs.bullet import CartPoleBulletEnv
from pybullet_envs.gym_locomotion_envs import HopperBulletEnv, Walker2DBulletEnv, HalfCheetahBulletEnv, AntBulletEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bridge import Bridge
from gym_rarl.envs.adv_cartpole import AdversarialCartPoleEnv
from gym_rarl.envs.adv_walkers import AdversarialWalker2DEnv, AdversarialHalfCheetahEnv, AdversarialHopperEnv, \
    AdversarialAntEnv
from gym_rarl.envs.rarl_env import ProtagonistRarlEnv, AdversarialRarlEnv

verbose = False
renders = False
ts = 300
seed = 123
env_pairs = [
    (HopperBulletEnv, AdversarialHopperEnv),
    (CartPoleBulletEnv, AdversarialCartPoleEnv),
    (Walker2DBulletEnv, AdversarialWalker2DEnv),
    (HalfCheetahBulletEnv, AdversarialHalfCheetahEnv),
    (AntBulletEnv, AdversarialAntEnv),
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


def init_adv(adv_env_c, disable_adv=False):
    bridge = Bridge()
    prot_env = make_vec_env(
        lambda: ProtagonistRarlEnv(adv_env_c(render=renders), bridge),
        seed=seed
    )
    adv_env = make_vec_env(
        lambda: AdversarialRarlEnv(adv_env_c(render=renders), bridge),
        seed=seed
    )
    prot_env = VecNormalize(prot_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    adv_env = VecNormalize(adv_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    prot_agent = PPO('MlpPolicy', prot_env, verbose=verbose, seed=seed, n_steps=ts)
    if disable_adv:
        bridge.link_agents(prot_agent, None)
    else:
        adv_agent = PPO('MlpPolicy', adv_env, verbose=verbose, seed=seed, n_steps=ts)
        bridge.link_agents(prot_agent, adv_agent)
    return prot_agent, prot_env


def init_control(env_c):
    env = make_vec_env(
        lambda: env_c(**{'renders' if env_c.__name__ == 'CartPoleBulletEnv' else 'render': renders}), seed=seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    agent = PPO('MlpPolicy', env, verbose=verbose, seed=seed, n_steps=ts)
    return agent, env


class RarlSanityTests(unittest.TestCase):
    def test_null_adversary_acts_the_same(self):
        for env_c, adv_env_c in env_pairs:
            print(env_c.__name__, adv_env_c.__name__)
            # do control
            agent, env = init_control(env_c)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary disengaged
            prot_agent, prot_env = init_adv(adv_env_c, disable_adv=True)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert np.array_equal(last_adv_obs, last_control_obs)

    def test_acting_adversary_acts_different(self):
        for env_c, adv_env_c in env_pairs:
            agent, env = init_control(env_c)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary engaged
            prot_agent, prot_env = init_adv(adv_env_c)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert not np.array_equal(last_adv_obs, last_control_obs)

    def test_different_mass_acts_different(self):
        for env_c, adv_env_c in env_pairs:
            agent, env = init_control(env_c)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary engaged
            prot_agent, prot_env = init_adv(partial(adv_env_c, mass_percentage=0.5), disable_adv=True)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert not np.array_equal(last_adv_obs, last_control_obs)

    def test_different_friction_acts_different(self):
        for env_c, adv_env_c in env_pairs:
            agent, env = init_control(env_c)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary engaged
            prot_agent, prot_env = init_adv(partial(adv_env_c, friction_percentage=0.5), disable_adv=True)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            if 'CartPole' not in adv_env_c.__name__:
                assert not np.array_equal(last_adv_obs, last_control_obs)


if __name__ == '__main__':
    unittest.main()
