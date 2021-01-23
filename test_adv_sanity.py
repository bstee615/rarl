import unittest

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
ts = 16
ep = 3
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
    initial_control_obs = env.reset()
    agent.learn(ts * ep)
    last_control_obs = env.reset()
    last_control_prediction, _ = agent.predict(last_control_obs)
    env.close()
    return initial_control_obs, last_control_obs, last_control_prediction


class RarlSanityTests(unittest.TestCase):
    def test_control_env_behaves_like_rarl_env(self):
        for env_c, adv_env_c in env_pairs:
            # do control
            env = make_vec_env(lambda: env_c(), seed=seed)
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            agent = PPO('MlpPolicy', env, verbose=verbose, seed=seed, n_steps=ts)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary disengaged
            bridge = Bridge()
            env = make_vec_env(
                lambda: ProtagonistRarlEnv(adv_env_c(), bridge),
                seed=seed
            )
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            agent = PPO('MlpPolicy', env, verbose=verbose, seed=seed, n_steps=ts)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(agent, env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert np.array_equal(last_adv_obs, last_control_obs)
            assert np.array_equal(last_adv_prediction, last_control_prediction)

    def test_control_env_doesnt_behave_like_rarl_env(self):
        for env_c, adv_env_c in env_pairs:
            env = make_vec_env(lambda: env_c(), seed=seed)
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            agent = PPO('MlpPolicy', env, verbose=verbose, seed=seed, n_steps=ts)
            initial_control_obs, last_control_obs, last_control_prediction = learn_love_log(agent, env)

            # do adv env with adversary engaged
            bridge = Bridge()
            prot_env = make_vec_env(
                lambda: ProtagonistRarlEnv(adv_env_c(), bridge),
                seed=seed
            )
            adv_env = make_vec_env(
                lambda: AdversarialRarlEnv(adv_env_c(), bridge),
                seed=seed
            )
            prot_env = VecNormalize(prot_env, norm_obs=True, norm_reward=True, clip_obs=10.)
            adv_env = VecNormalize(adv_env, norm_obs=True, norm_reward=True, clip_obs=10.)
            prot_agent = PPO('MlpPolicy', prot_env, verbose=verbose, seed=seed, n_steps=ts)
            adv_agent = PPO('MlpPolicy', adv_env, verbose=verbose, seed=seed, n_steps=ts)
            bridge.link_agents(prot_agent, adv_agent)
            initial_adv_obs, last_adv_obs, last_adv_prediction = learn_love_log(prot_agent, prot_env)

            assert np.array_equal(initial_adv_obs, initial_control_obs)
            assert not np.array_equal(last_adv_obs, last_control_obs)
            assert not np.array_equal(last_adv_prediction, last_control_prediction)


if __name__ == '__main__':
    unittest.main()
