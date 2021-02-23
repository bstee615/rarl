import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker


def get_param(r, p):
    return next((a.split('=')[-1] for a in r["args"] if p in a), None)


def register(result):
    print(result["args"], result["avg_reward"], result["std_reward"])
    name = get_param(result, '--name')
    adversarial_agent = get_param(result, '--control') is None
    adversarial_env = get_param(result, '--force-no-adversarial') is None
    rewards = np.array(result["rewards"])
    return adversarial_agent, adversarial_env, rewards


def plot_results(results, env):
    max_reward = max((max(r["rewards"]) for r in results))
    for (ar, cr) in zip(results[0::2], results[1::2]):
        first_adv_agent, first_adv_env, first_rewards = register(ar)
        second_adv_agent, second_adv_env, second_rewards = register(cr)
        assert first_adv_env == second_adv_env
        filtered = [[
            i,
            np.percentile(first_rewards, i),
            np.percentile(second_rewards, i),
        ] for i in range(0, 101)]
        percentiles = np.array(filtered)
        data = pd.DataFrame(percentiles, columns=['Percentile', 'RARL', 'Control'])
        data2 = pd.melt(data, 'Percentile', value_name='Reward', var_name='Agent')
        ax = sns.lineplot(x='Percentile', y='Reward', hue='Agent', ci=None, data=data2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        plt.ylim([0, max_reward])
        figname = f"{env} {'Adversarial' if second_adv_env else 'Control'} environment"
        plt.title(figname)
        plt.savefig(f'plots/{figname}')
        plt.show()


for envname in ['Ant', 'HalfCheetah']:
    env = f'Adversarial{envname}BulletEnv-v0'
    with open(f'logs/eval-{env}/summary.pkl', 'rb') as f:
        results = pickle.load(f)
    plot_results(results, envname)
