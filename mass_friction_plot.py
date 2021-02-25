import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_param(r, p):
    return next((a.split('=')[-1] for a in r["args"] if p in a), None)


def register(result, mass_friction):
    print(result["args"], result["avg_reward"], result["std_reward"])
    name = get_param(result, '--name')
    adversarial_agent = get_param(result, '--control') is None
    rewards = np.array(result["rewards"])
    arg_percentage = float(get_param(result, f'--{mass_friction}_percentage'))
    return adversarial_agent, arg_percentage, rewards


def plot_results(results, env):
    zipped = zip(results[0::2], results[1::2])
    for mass_friction in ['mass', 'friction']:
        results_and_percents = []
        for percentage in np.arange(0.5, 1.5, 0.1):
            results_and_percents.append((percentage, next(zipped)))
        max_reward = max(max(r["rewards"] + c["rewards"]) for _, (r, c) in results_and_percents)
        min_reward = min(min(r["rewards"] + c["rewards"]) for _, (r, c) in results_and_percents)
        percentage_rewards = []
        for percentage, (adv_results, control_results) in results_and_percents:
            percentage = round(percentage, 1)
            first_adv_agent, rarl_arg_percentage, first_rewards = register(adv_results, mass_friction)
            second_adv_agent, control_arg_percentage, second_rewards = register(control_results, mass_friction)
            assert (rarl_arg_percentage - percentage) ** 2 < 0.1
            assert (control_arg_percentage - percentage) ** 2 < 0.1
            assert len(first_rewards) == len(second_rewards)
            assert first_adv_agent
            assert not second_adv_agent
            for i in range(len(first_rewards)):
                percentage_rewards.append([percentage, first_rewards[i], second_rewards[i]])
        percentage_rewards = np.array(percentage_rewards)
        data = pd.DataFrame(percentage_rewards, columns=[f'{mass_friction} %', 'RARL', 'Control'])
        data2 = pd.melt(data, f'{mass_friction} %', value_name='Reward', var_name='Agent')
        ax = sns.lineplot(x=f'{mass_friction} %', y='Reward', hue='Agent', ci='sd', data=data2)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        plt.ylim([min_reward, max_reward])
        figname = f"{env} with varying {mass_friction}"
        plt.title(figname)
        plt.savefig(f'plots/{figname}')
        plt.show()


for envname in ['Ant', 'Hopper', 'HalfCheetah']:
    env = f'Adversarial{envname}BulletEnv-v0'
    with open(f'logs/eval-mass-friction-{env}/summary.pkl', 'rb') as f:
        results = pickle.load(f)
    plot_results(results, envname)
