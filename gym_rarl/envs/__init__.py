import gym
from gym.envs.registration import registry


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# Adversarial Bullet envs

register(
    id='AdversarialCartPoleBulletEnv-v0',
    entry_point='gym_rarl.envs.adv_cartpole:AdversarialCartPoleBulletEnv',
    max_episode_steps=200,
    reward_threshold=190.0,
)

# register(
#     id='AdversarialCartPoleContinuousBulletEnv-v0',
#     entry_point='gym_rarl.envs.adv_cartpole:AdversarialCartPoleContinuousBulletEnv',
#     max_episode_steps=200,
#     reward_threshold=190.0,
# )

register(id='AdversarialWalker2DBulletEnv-v0',
         entry_point='gym_rarl.envs.adv_walkers:AdversarialWalker2DBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='AdversarialHalfCheetahBulletEnv-v0',
         entry_point='gym_rarl.envs.adv_walkers:AdversarialHalfCheetahBulletEnv',
         max_episode_steps=1000,
         reward_threshold=3000.0)

register(id='AdversarialAntBulletEnv-v0',
         entry_point='gym_rarl.envs.adv_walkers:AdversarialAntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='AdversarialHopperBulletEnv-v0',
         entry_point='gym_rarl.envs.adv_walkers:AdversarialHopperBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)


def getList():
    btenvs = [spec.id for spec in gym.envs.registry.all() if spec.id.find('Adversarial') >= 0]
    return btenvs
