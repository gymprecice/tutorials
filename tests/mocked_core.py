from abc import ABC, abstractmethod
from os import getcwd, path

import gymnasium as gym


class Adapter(ABC, gym.Env):
    """
    Mock class to represent gymprecice 'Adapter' in all environment tests.
    """

    def __init__(self, options, *args) -> None:
        self._solver_list = options["physics_simulation_engine"]["solvers"]
        self._solver_list = options["physics_simulation_engine"]["solvers"]
        self._reset_script = options["physics_simulation_engine"]["reset_script"]
        self._prerun_script = options["physics_simulation_engine"].get(
            "prerun_script", self._reset_script
        )
        self._run_script = options["physics_simulation_engine"]["run_script"]
        self._controller_config = options["controller"]

        self._env_dir = "env_0"
        self._env_path = path.join(getcwd(), self._env_dir)
        self._t = 0
        self._dt = 0.0005
        self._time_window = 0

    def reset(self, *args):
        raise NotImplementedError

    def step(self, *args):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def _set_precice_vectices(self, *args):
        raise NotImplementedError

    def _init_precice(self):
        raise NotImplementedError

    def _advance(self, *args):
        raise NotImplementedError

    def _write(self, *args):
        raise NotImplementedError

    def _launch_subprocess(self, *args):
        raise NotImplementedError

    def _check_subprocess_exists(self, *args):
        raise NotImplementedError

    def _finalize_subprocess(self, *args):
        raise NotImplementedError

    def _dummy_episode(self):
        raise NotImplementedError

    def _finalize(self):
        raise NotImplementedError

    @abstractmethod
    def _get_action(self, *args):
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError

    @abstractmethod
    def _close_external_resources(self):
        pass

    def __del__(self):
        pass
