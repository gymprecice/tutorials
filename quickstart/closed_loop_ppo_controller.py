import gymnasium as gym

from gymprecice.utils.fileutils import make_result_dir

from environment import JetCylinder2DEnv

import torch
import torch.nn as nn

import numpy as np
import math
from os import path, getcwd, system


class Controller(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_actions = np.prod(env.action_space.shape)
        self.n_obs = np.prod(env.observation_space.shape)
        self.action_scale = (env.action_space.high - env.action_space.low) / 2.0
        self.action_bias = (env.action_space.high + env.action_space.low) / 2.0

        self.actor = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_actions),
        )

    def act(self, obs):
        obs = obs.reshape(-1, self.n_obs)
        unbounded_action = self.actor(torch.Tensor(obs).to("cpu"))
        # bound action
        squashed_action = torch.tanh(unbounded_action)
        action = squashed_action * self.action_scale + self.action_bias
        return torch.flatten(action, start_dim=0)


if __name__ == "__main__":
    current_path = getcwd()
    
    # make the environment
    environment_config = make_result_dir()
    env = JetCylinder2DEnv(environment_config, 0)
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    env = gym.wrappers.ClipAction(env)
    terminated = False

    # create the controller and load it with a pre-trained network
    system(f"cp {current_path}/trained_agent.pt {getcwd()}")
    controller = Controller(env)
    # load the controller with a pre-trained Reinforcement Learning agent
    controller.load_state_dict(torch.load("trained_agent.pt"), strict=False)
    controller.eval()

    # reset the environment
    obs, info = env.reset()

    # step through the environment and control it for one complete episode
    while not terminated:
        with torch.no_grad():
            action = controller.act(obs)
        obs, _, terminated, _, _ = env.step(action)

    # close the environment
    env.close()
