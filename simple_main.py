import gym
from ray import tune
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLogger
from simple_trainer import GATrainer
from ray.rllib.env.atari_wrappers import NoopResetEnv, WarpFrame, FrameStack


def get_env():
    env = gym.make('FrostbiteDeterministic-v4')
    env = NoopResetEnv(env, noop_max=30)
    env = WarpFrame(env, 84)
    env = FrameStack(env, 4)
    return env

register_env("frostbite", get_env)

config = {
    "env": "frostbite",
    "logger_config": {
        "wandb": {
            "project": "deep-neuroevolution",
            "api_key": "~"
        },
    },
    "log_level": "ERROR",
    "num_gpus": 0,
    "num_workers": 31,
    "population_size": 1000,
    "max_timesteps_per_episode": 5000,
    "mutation_power": 0.005,
}

tune.run(
    GATrainer,
    name="GA",
    stop={"timesteps_total": 500_000_000},
    loggers=[WandbLogger],
    config=config
)
