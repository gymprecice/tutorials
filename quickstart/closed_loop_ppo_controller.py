from os import getcwd, system

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from environment import JetCylinder2DEnv
from gymprecice.utils.fileutils import make_result_dir


class Controller(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_actions = np.prod(env.action_space.shape)
        self.n_obs = np.prod(env.observation_space.shape)
        self.action_scale = (env.action_space.high - env.action_space.low) / 2.0
        self.action_bias = (env.action_space.high + env.action_space.low) / 2.0

        self.actor = nn.Sequential(
            nn.Linear(self.n_obs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, self.n_actions)

    def act(self, obs):
        obs = obs.reshape(-1, self.n_obs)
        latent_pi = self.actor(torch.Tensor(obs).to("cpu"))
        unscaled_action = torch.tanh(self.actor_mean(latent_pi))
        action = unscaled_action * self.action_scale + self.action_bias
        return torch.flatten(action, start_dim=0)


if __name__ == "__main__":
    current_path = getcwd()

    # make the environment
    environment_config = make_result_dir(time_stamped=False, suffix="ppo")
    env = JetCylinder2DEnv(environment_config, 0)
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    env.latest_available_sim_time = 0
    env.action_interval = 25
    terminated = False

    # create the controller and load it with a pre-trained network
    system(f"cp {current_path}/trained_agent.pt {getcwd()}")
    controller = Controller(env)
    controller.load_state_dict(torch.load("trained_agent.pt"), strict=False)
    controller.eval()

    # reset the environment
    obs, info = env.reset()

    print("\n...")
    print("The PPO control case is running!")
    print(
        "This task is expected to be completed in about 5 minutes on a system with two cores @ 2.10GHz."
    )
    print("...\n")

    # step through the environment and control it for one complete episode (8 seconds, 320 steps)
    while not terminated:
        with torch.no_grad():
            action = controller.act(obs)
        obs, _, terminated, _, _ = env.step(action)

    print("The control case is done.")

    # close the environment
    env.close()
