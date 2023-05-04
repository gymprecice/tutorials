"""AFC environment for perpendicular-flap control.

The physics-simulation-engine used for this environment is a fluid-structure interaction model adapted
from ["preCICE tutorials"](https://github.com/precice/tutorials/tree/master/perpendicular-flap) under the following licence:

[GNU LESSER GENERAL PUBLIC LICENSE](https://github.com/precice/tutorials/blob/master/LICENSE)
"""

import logging
import math
from os.path import join

import gymnasium as gym
import numpy as np
from gymprecice.core import Adapter
from gymprecice.utils.fileutils import open_file
from gymprecice.utils.openfoamutils import (
    get_interface_patches,
    get_patch_geometry,
    read_line,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class PerpendicularFlapEnv(Adapter):
    r"""An AFC environment to perform open-loop control for a wall-mounted elastic flap in a two-dimensional channel flow.

    ## Description
    An environment to showcase the modularity of Gym-preCICE adapter to couple a controller to a multi-solver physics-simulation-engine (more than one PDE solver
    from different simulation software packages).
    We control centre position of an inflow jet to manipulate the motion of a wall-mounted elastic flap in a two-dimensional channel flow.
    The goal is to keep the elastic flap oscillating within the channel flow.

    ## Action Space
    The action is a `ndarray` with shape `(1,)` with the value corresponding to the centre position of the inflow jet.
    **Note**: Each control action (centre position of the inflow jet) is repeated over `self.action_interval` simulation time steps.

    ## Observation Space
    The observation is a `ndarray` with shape `(33, 2)` with the values corresponding to 2D force vectors acting on the elastic flap.

    ## Rewards
    Since we do not use this environment to train a DRL agent, we do not need to define a reward signal.

    ## Episode End
    The episode ends after 50 seconds of fluid-structure interaction simulation.

    Args:
        Adapter: gymprecice adapter super-class
    """

    def __init__(self, options: dict = None, idx: int = 0) -> None:
        """Environment constructor.

        Args:
            options: a dictionary containing the information within gymprecice-config.json. It is a return of `gymprecice.utils.fileutils.make_result_dir` method called within the controller algorithm.
            idx: environment index.
        """
        super().__init__(options, idx)
        self._inlet_height = 4
        self._inlet_max_velocity = 10
        self._Q_ref = 2.0 / 3.0 * self._inlet_max_velocity * self._inlet_height

        self._jet_height = self._inlet_height / 2.0
        self._jet_max_velocity = (
            self._inlet_height / self._jet_height * self._inlet_max_velocity
        )

        self._n_probes = 33  # number of flap-patch faces
        self.action_interval = 10
        self.reward_average_time_window = 0.1

        self.action_space = gym.spaces.Box(
            low=self._jet_height / 2.0,
            high=self._inlet_height - self._jet_height / 2.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_probes, 2), dtype=np.float32
        )

        self._reward_info = {
            "filed_name": "displacement",
            "n_columns": 6,  # number of data columns (excluding the time column)
            "displacement_column": 4,
            "file_path": "precice-Solid-watchpoint-Flap-Tip.log",
            "file_handler": None,
            "data": None,  # live data for the controlled period (t > self._latest_available_sim_time)
        }

        # find openfoam solver (we have only one openfoam solver)
        openfoam_case_name = ""
        dealii_case_name = ""
        for solver_name in self._solver_list:
            if solver_name.rpartition("-")[-1].lower() == "openfoam":
                openfoam_case_name = solver_name
            elif solver_name.rpartition("-")[-1].lower() == "dealii":
                dealii_case_name = solver_name
        self._openfoam_solver_path = join(self._env_path, openfoam_case_name)
        self._dealii_solver_path = join(self._env_path, dealii_case_name)

        openfoam_interface_patches = get_interface_patches(
            join(openfoam_case_name, "system", "preciceDict")
        )

        action_patch = []
        self.action_patch_geometric_data = {}
        for interface in self._controller_config["write_to"]:
            if interface in openfoam_interface_patches:
                action_patch.append(interface)
        self.action_patch_geometric_data = get_patch_geometry(
            openfoam_case_name, action_patch
        )
        action_patch_coords = {}
        for patch_name in self.action_patch_geometric_data.keys():
            action_patch_coords[patch_name] = [
                np.delete(coord, 2)
                for coord in self.action_patch_geometric_data[patch_name]["face_centre"]
            ]

        observation_patch = []
        for interface in self._controller_config["read_from"]:
            if interface in openfoam_interface_patches:
                observation_patch.append(interface)
        self.observation_patch_geometric_data = get_patch_geometry(
            openfoam_case_name, observation_patch
        )
        observation_patch_coords = {}
        for patch_name in self.observation_patch_geometric_data.keys():
            observation_patch_coords[patch_name] = [
                np.delete(coord, 2)
                for coord in self.observation_patch_geometric_data[patch_name][
                    "face_centre"
                ]
            ]

        patch_coords = {
            "read_from": observation_patch_coords,
            "write_to": action_patch_coords,
        }

        self._set_precice_vectices(patch_coords)

    @property
    def inlet_max_velocity(self):
        """Get the maximum velocity of the inlet jet."""
        return self._inlet_max_velocity

    @inlet_max_velocity.setter
    def inlet_max_velocity(self, value):
        """Set the maximum velocity of the inlet jet."""
        self._inlet_max_velocity = value
        self._Q_ref = 2.0 / 3.0 * value * self._inlet_height
        self._jet_max_velocity = self._inlet_height / self._jet_height * value

    @property
    def jet_height(self):
        """Get the height of the inlet jet."""
        return self._jet_height

    @jet_height.setter
    def jet_height(self, value):
        """Set the height of the inlet jet."""
        self._jet_height = value
        self._jet_max_velocity = self._inlet_height / value * self._inlet_max_velocity
        self.action_space = gym.spaces.Box(
            low=value / 2,
            high=self._inlet_height - value / 2,
            shape=(1,),
            dtype=np.float32,
        )

    def step(self, action):
        r"""Repeat the control action over `self.action_interval` simulation time steps."""
        return self._repeat_step(action)

    def _get_action(self, action, write_var_list):
        acuation_interface_field = self._action_to_patch_field(action)
        write_data = {
            var: acuation_interface_field[var.rpartition("-")[-1]]
            for var in write_var_list
        }
        return write_data

    def _get_observation(self, observation_interface_field, read_var_list):
        observation = []
        for var in read_var_list:
            observation.append(observation_interface_field[var])
        return np.array(observation).squeeze()

    def _get_reward(self):
        return self._displacement_to_reward()

    def _close_external_resources(self):
        # close probes and forces files
        try:
            if self._reward_info["file_handler"] is not None:
                self._reward_info["file_handler"].close()
                self._reward_info["file_handler"] = None
        except Exception as err:
            logger.error("Can't close probes/forces file")
            raise err

    def _action_to_patch_field(self, action):
        jet_centre_position = action[0]

        # velocity field of the actuation patches
        U_profile = {}
        for patch_name in self.action_patch_geometric_data.keys():
            Cf = self.action_patch_geometric_data[patch_name]["face_centre"]
            Sf = self.action_patch_geometric_data[patch_name]["face_area_vector"]

            U_patch = []
            uf = []
            inlet_bound = [
                jet_centre_position - self._jet_height / 2.0,
                jet_centre_position + self._jet_height / 2.0,
            ]
            for c in Cf:
                if c[1] < inlet_bound[0] or c[1] > inlet_bound[1]:
                    # inactive inlet
                    U_patch.append(np.zeros(3))
                    uf.append(np.zeros(3))
                else:
                    # active parabolic inlet
                    y = c[1] - inlet_bound[0]
                    U_patch.append(
                        np.array(
                            [
                                4
                                * self._jet_max_velocity
                                * y
                                * (self._jet_height - y)
                                / (self._jet_height**2),
                                0.0,
                                0.0,
                            ]
                        )
                    )
                    uf.append(np.array([1.0, 0.0, 0.0]))

            Q_calc = -sum([u.dot(s) for u, s in zip(U_patch, Sf)])
            active_area = -sum([s.dot(n) for s, n in zip(Sf, uf)])
            # correct velocity profile to enforce fixed mass-flux
            Q_err = self._Q_ref - Q_calc
            U_err = Q_err / active_area * np.array(uf)
            U_patch += U_err

            # return the velocity profile
            Q_final = -sum([u.dot(s) for u, s in zip(U_patch, Sf)])

            if np.isclose(Q_final, self._Q_ref):
                U_profile[patch_name] = np.array(
                    [np.delete(item, 2) for item in U_patch]
                )
            else:
                raise Exception("Not a synthetic jet: Q_jet1 + Q_jet2 is not zero")

        return U_profile

    def _displacement_to_reward(self):
        self._read_displacement_from_file()

        assert self._reward_info["data"], "displacement-data is empty!"
        displacement_data = self._reward_info["data"]

        n_lookback = int(self.reward_average_time_window // self._dt)

        # avoid the starting again and again from t0 by working in reverse order
        reversed_displacement_data = displacement_data[::-1]
        if len(reversed_displacement_data) > n_lookback:
            tip_displacement = np.array(
                [
                    reversed_displacement_data[idx][2][
                        self._reward_info["displacement_column"] - 1
                    ]
                    for idx in range(n_lookback)
                ]
            )
        else:
            tip_displacement = np.array(
                [
                    data_line[2][self._reward_info["displacement_column"] - 1]
                    for data_line in reversed_displacement_data
                ]
            )

        reward = np.sum(np.abs(np.diff(tip_displacement)))
        return reward

    def _read_displacement_from_file(self):
        # sequential read of a single line (last line) of tip-displacement file at each RL step

        data_path = join(self._dealii_solver_path, self._reward_info["file_path"])

        if self._reward_info["file_handler"] is None:
            file_object = open_file(data_path)
            self._reward_info["file_handler"] = file_object
            self._reward_info["data"] = []

        logger.debug(f"reading forces from: {data_path}")

        new_time_stamp = True
        latest_time_stamp = self._t - self._dt
        if self._reward_info["data"]:
            new_time_stamp = self._reward_info["data"][-1][0] != latest_time_stamp

        if new_time_stamp:
            time_stamp = 0
            while not math.isclose(
                time_stamp, latest_time_stamp
            ):  # read till the end of a time-window
                while True:
                    is_comment, time_stamp, n_forces, forces_data = read_line(
                        self._reward_info["file_handler"],
                        self._reward_info["n_columns"],
                    )
                    if not is_comment and n_forces == self._reward_info["n_columns"]:
                        break
                self._reward_info["data"].append([time_stamp, n_forces, forces_data])
            assert math.isclose(
                time_stamp, latest_time_stamp
            ), f"Mismatched time data: {time_stamp} vs {self._t}"

    def _repeat_step(self, action):
        subcycle = 0
        while subcycle < self.action_interval:
            if isinstance(action, np.ndarray):
                next_obs, reward, terminated, truncated, info = super().step(action)
            else:
                next_obs, reward, terminated, truncated, info = super().step(
                    action.cpu().numpy()
                )

            subcycle += 1
            if terminated or truncated:
                break

        return next_obs, reward, terminated, truncated, info
