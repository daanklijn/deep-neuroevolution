import logging
import numpy as np
import ray
from ray.rllib.agents import Trainer, with_common_config
from ray.rllib.utils.annotations import override
import tensorflow as tf
from simple_worker import Worker

tf.compat.v1.enable_eager_execution()

DEFAULT_CONFIG = with_common_config({
    "population_size": 1000,
    "max_timesteps_per_episode": 2000,
    "max_evaluation_steps": 2000,
    "number_elites": 20,
    "mutation_power": 0.002,
    "num_workers": 7,
})


class GATrainer(Trainer):
    _name = "GA"

    @override(Trainer)
    def _init(self, config, env_creator):
        self.config = config
        self._workers = [
            Worker.remote(config, env_creator)
            for _ in range(config["num_workers"])
        ]

        self.episodes_total = 0
        self.timesteps_total = 0
        self.generation = 0
        self.elite_weights = []

    @override(Trainer)
    def step(self):
        worker_jobs = []
        for i in range(self.config['population_size']):
            elite_id = i % self.config['number_elites']
            worker_id = i % self.config['num_workers']
            weights = self.elites[elite_id] if self.elite_weights else None
            worker_jobs += [self._workers[worker_id].evaluate.remote(weights, True, False)]

        results = ray.get(worker_jobs)
        rewards = [result['total_reward'] for result in results]
        elites = np.argsort(rewards)[-self.config['number_elites']:]

        self.elites = []
        for result_id in elites:
            self.elites.append(results[result_id]['weights'])

        self.timesteps_total += sum([result['timesteps_total'] for result in results])
        self.episodes_total += len(results)
        self.generation += 1

        return dict(
            timesteps_total=self.timesteps_total,
            episodes_total=self.episodes_total,
            generation=self.generation,
            train_reward_min=np.min(rewards),
            train_reward_mean=np.mean(rewards),
            train_reward_med=np.median(rewards),
            train_reward_max=np.max(rewards),
        )

