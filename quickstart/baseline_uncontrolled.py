import gymnasium as gym
import torch
from environment import JetCylinder2DEnv
from gymprecice.utils.fileutils import make_result_dir


class Controller:
    """A dymmy controller with no action."""

    def act(self):
        return torch.zeros(1)


if __name__ == "__main__":
    # make the environment
    environment_config = make_result_dir(time_stamped=False, suffix="baseline")
    env = JetCylinder2DEnv(environment_config, 0)
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    env.latest_available_sim_time = 0
    env.action_interval = 10
    terminated = False

    # create the sinusoidal controller
    controller = Controller()

    # reset the environment
    _, _ = env.reset()

    print("\n...")
    print("The baseline case (no-controll) is running!")
    print(
        "This task is expected to be completed in about 5 minutes on a system with two cores @ 2.10GHz."
    )
    print("...\n")

    # step through the environment and control it for one complete episode (8 seconds, 320 steps)
    while not terminated:
        action = controller.act()
        _, _, terminated, _, _ = env.step(action)

    print("The baseline case is done.")

    # close the environment
    env.close()
