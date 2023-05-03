import math

import gymnasium as gym
from environment import JetCylinder2DEnv
from gymprecice.utils.fileutils import make_result_dir


class Controller:
    def __init__(self, env):
        self._t = 0.0
        self._dt = 0.025
        self.action_scale = (env.action_space.high - env.action_space.low) / 2.0
        self.action_bias = (env.action_space.high + env.action_space.low) / 2.0

    def act(self):
        unscaled_action = math.sin(4 * math.pi * self._t)
        action = unscaled_action * self.action_scale + self.action_bias
        self._t += self._dt
        return action


if __name__ == "__main__":
    # make the environment
    environment_config = make_result_dir(time_stamped=False, suffix="sinusoidal")
    env = JetCylinder2DEnv(environment_config, 0)
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    env.latest_available_sim_time = 0
    env.action_interval = 25
    terminated = False

    # create the sinusoidal controller
    controller = Controller(env)

    # reset the environment
    _, _ = env.reset()

    print("\n...")
    print("The control case is running!")
    print(
        "This task is expected to be completed in about 5 minutes on a system with two cores @ 2.10GHz."
    )
    print("...\n")

    # step through the environment and control it for one complete episode (8 seconds, 320 steps)
    while not terminated:
        action = controller.act()
        _, _, terminated, _, _ = env.step(action)

    print("The control case is done.")

    # close the environment
    env.close()
