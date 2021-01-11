import ray
import wandb
from gym.wrappers.monitoring import video_recorder
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.tune.integration.wandb import wandb_mixin

from utils.atari_model import AtariModel
from utils.chromosome import Chromosome
import numpy as np


@ray.remote
class Worker:
    def __init__(self,
                 config,
                 env_creator):
        self.maximum_timesteps = config['max_timesteps_per_episode']
        self.mutation_power = config['mutation_power']
        self.model = AtariModel()
        self.env = env_creator({})

    def evaluate(self, weights, mutate, record):
        if weights:
            self.model.set_weights(weights)
        if mutate:
            self.model.mutate(self.mutation_power)

        obs = self.env.reset()
        rewards = []

        for ts in range(self.maximum_timesteps):
            action = self.model.determine_actions(np.array([obs]))
            obs, reward, done, info = self.env.step(action)
            rewards += [reward]
            if done:
                break

        return {
            'total_reward': sum(rewards),
            'timesteps_total': ts,
            'weights': self.model.get_weights(),
        }
