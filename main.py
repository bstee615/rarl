import logging

from stable_baselines3 import PPO
# Set up environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from arguments import parse_args
from bridge import Bridge


def setup(args):
    bridge = Bridge()

    render_key = "renders" if 'CartPole' in args.env else "render"
    env_kwargs = {
        render_key: args.render,
        "adv_force": args.adv_force,
        "mass_percentage": args.mass_percentage,
        "friction_percentage": args.friction_percentage,
        "simple_reward": args.simple_reward,
    }

    env = make_vec_env(args.env, env_kwargs=env_kwargs, seed=args.seed, monitor_dir=args.monitor_dir)

    if args.evaluate:
        env = VecNormalize.load(f'{args.pickle}-{args.envname}', env)
        prot_agent = PPO.load(f'{args.pickle}-{args.prot_name}')
        if prot_agent.seed != args.seed:
            logging.info(f'warning: {prot_agent.seed=} does not match { args.seed=}')

        if args.adversarial:
            adv_agent = PPO.load(args.adv_pickle)
            if adv_agent.seed != args.seed:
                logging.info(f'warning: {adv_agent.seed=} does not match { args.seed=}')
        else:
            adv_agent = None
    else:
        env = VecNormalize(env)
        prot_logname = f'{args.logs}-{args.prot_name}' if args.logs else None
        prot_agent = PPO("MlpPolicy", env, verbose=args.verbose, seed=args.seed,
                         tensorboard_log=prot_logname, n_steps=args.N_steps, is_protagonist=True, bridge=bridge)

        if args.adversarial:
            adv_logname = f'{args.logs}-{args.adv_name}' if args.logs else None
            adv_agent = PPO("MlpPolicy", env, verbose=args.verbose, seed=args.seed,
                            tensorboard_log=adv_logname, n_steps=args.N_steps, is_protagonist=False, bridge=bridge)
        else:
            adv_agent = None

    bridge.link_agents(prot_agent, adv_agent)

    return prot_agent, adv_agent, env


def run(args, evaluate_fn=None):
    prot, adv, env = setup(args)
    try:
        if args.evaluate:
            env.training = False
            env.norm_reward = False
            reward, lengths = evaluate_policy(prot, env, args.N_eval_episodes,
                                              return_episode_rewards=True, adversarial=adv if adv else None)
            return reward
        else:
            # Train
            """
            Train according to Algorithm 1
            """
            steps_done = 0
            for i in range(args.N_iter):
                # Do N_mu rollouts training the protagonist
                prot.learn(total_timesteps=args.N_mu * args.N_steps, reset_num_timesteps=i == 0)
                steps_done += args.N_mu * args.N_steps
                # Evaluate protagonist reward or whatever
                if evaluate_fn is not None:
                    evaluate_fn(steps_done)
                # Do N_nu rollouts training the adversary
                if adv is not None:
                    adv.learn(total_timesteps=args.N_nu * args.N_steps, reset_num_timesteps=i == 0)
                if args.save_every is not None:
                    if steps_done % args.save_every == 0:
                        logging.info(f'saving at {steps_done=}...')
                        prot.save(f'{args.pickle}-{args.prot_name}')
                        env.save(f'{args.pickle}-{args.envname}')

            prot.save(f'{args.pickle}-{args.prot_name}')
            env.save(f'{args.pickle}-{args.envname}')

            if adv is not None:
                adv.save(args.adv_pickle)
    finally:
        env.close()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    result = run(args)
    if result is not None:
        avg_reward, std_reward = result
        logging.info(f'reward={avg_reward}+={std_reward}')


if __name__ == '__main__':
    main()
