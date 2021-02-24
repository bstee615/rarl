from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


def objective(x, a, b):
    return a * (x ** 0.5) + b


def trainable(config):
    # config (dict): A dict of hyperparameters.

    for x in range(20):
        score = objective(x, config["a"], config["b"])

        tune.report(score=score)  # This sends the score to Tune.


search = HyperOptSearch()
sched = ASHAScheduler()

# Pass in a Trainable class or function to tune.run.
anal = tune.run(trainable,
                config={
                    "a": tune.uniform(0, 1),
                    "b": tune.uniform(0, 1)
                },
                num_samples=10,
                scheduler=sched,
                search_alg=search,
                metric="score",
                mode="max")

best_trial = anal.best_trial
best_config = anal.best_config
best_logdir = anal.best_logdir
best_result = anal.best_result
best_result_df = anal.best_result_df
