### a work-around to find teh parent package (gymprecice-tutorials) without any installation setup
import sys
from os import getcwd
sys.path.append(getcwd())
###

import pytest
from pytest_mock import mocker, class_mocker

from shutil import rmtree
from os import chdir, path, makedirs, system

import numpy as np
import math

from . import mocked_core
from . import mocked_precice


@pytest.fixture
def testdir(tmpdir):
    test_dir = tmpdir.mkdir("test-jet-cylinder")
    yield chdir(test_dir)
    rmtree(test_dir)


@pytest.fixture
def patch_env_helpers(mocker):
    mocker.patch(
        "closed_loop_AFC.jet_control_cylinder.environment.get_interface_patches",
        return_value=[],
    )
    mocker.patch(
        "closed_loop_AFC.jet_control_cylinder.environment.get_patch_geometry",
        return_value={},
    )


@pytest.fixture(scope="class")
def mock_adapter(class_mocker):
    class_mocker.patch.dict("sys.modules", {"gymprecice.core": mocked_core})
    from gymprecice.core import Adapter

    Adapter.reset = class_mocker.MagicMock()
    Adapter.step = class_mocker.MagicMock()
    Adapter.close = class_mocker.MagicMock()
    Adapter._set_precice_vectices = class_mocker.MagicMock()
    Adapter._init_precice = class_mocker.MagicMock()
    Adapter._advance = class_mocker.MagicMock()
    Adapter._write = class_mocker.MagicMock()
    Adapter._launch_subprocess = class_mocker.MagicMock()
    Adapter._check_subprocess_exists = class_mocker.MagicMock()
    Adapter._finalize_subprocess = class_mocker.MagicMock()
    Adapter._dummy_episode = class_mocker.MagicMock()
    Adapter._finalize = class_mocker.MagicMock()
    Adapter._get_action = class_mocker.MagicMock()
    Adapter._get_observation = class_mocker.MagicMock()
    Adapter._get_reward = class_mocker.MagicMock()


@pytest.fixture(scope="class")
def mock_precice(class_mocker):
    class_mocker.patch.dict("sys.modules", {"precice": mocked_precice})


ENVIRONMENT_CONFIG = {
    "environment": {
        "name": "rotating_cylinder_2d"
    },
    "solvers": {
        "name": ["fluid-openfoam"],
        "reset_script": "clean.sh",
        "run_script": "run.sh"
    },
    "actuators": {
        "name": ["cylinder"]
    }
}


class TestJetCylinder2D:
    gymprecice_tutorials_dir = getcwd()
    def make_env(self):
        from closed_loop_AFC.jet_control_cylinder.environment import JetCylinder2DEnv
        JetCylinder2DEnv.__bases__ = (mocked_core.Adapter,)

        return JetCylinder2DEnv(ENVIRONMENT_CONFIG)

    def test_base(self, testdir, mock_precice):
        from closed_loop_AFC.jet_control_cylinder.environment import JetCylinder2DEnv
        assert JetCylinder2DEnv.__base__.__name__ == mocked_core.Adapter.__name__

    def test_base(self, testdir, mock_precice):
        from closed_loop_AFC.jet_control_cylinder.environment import JetCylinder2DEnv
        assert JetCylinder2DEnv.__base__.__name__ == mocked_core.Adapter.__name__

    def test_setters(self, testdir, patch_env_helpers, mock_adapter):
        n_probes = 10
        n_forces = 4
        min_jet_rate = -1
        max_jet_rate = 1
        env = self.make_env()
        env.n_probes = n_probes
        env.n_forces = n_forces
        env.min_jet_rate = min_jet_rate
        env.max_jet_rate = max_jet_rate

        check = {
            "n_of_probes": env._observation_info["n_probes"] == n_probes,
            "n_of_forces": env._reward_info["n_forces"] == n_forces,
            "action_space": (
                env.action_space.high == max_jet_rate
                and env.action_space.low == min_jet_rate
            ),
            "obs_space": env.observation_space.shape == (n_probes,),
        }
        assert all(check.values())

    @pytest.mark.parametrize(
        "input, expected",
        [
            (
                0,
                [
                    f"/postProcessing/probes/0/p",
                    f"/postProcessing/forceCoeffs/0/coefficient.dat",
                    False,
                ],
            ),
            (
                0.0,
                [
                    f"/postProcessing/probes/0/p",
                    f"/postProcessing/forceCoeffs/0/coefficient.dat",
                    False,
                ],
            ),
            (
                0.25,
                [
                    f"/postProcessing/probes/0.25/p",
                    f"/postProcessing/forceCoeffs/0.25/coefficient.dat",
                    True,
                ],
            ),
        ],
    )
    def test_latest_time(
        self, testdir, patch_env_helpers, mock_adapter, input, expected
    ):
        env = self.make_env()
        env.latest_available_sim_time = input

        check = {
            "obs_time_dir": env._observation_info["file_path"] == expected[0],
            "reward_time_dir": env._reward_info["file_path"] == expected[1],
            "prerun_data_required": env._prerun_data_required == expected[2],
        }
        assert all(check.values())

    def test_get_observation(self, testdir, patch_env_helpers, mock_adapter):
        latest_available_sim_time = 0.335
        n_probes = 5

        path_to_probes_dir = path.join(
            getcwd(), f"postProcessing/probes/{latest_available_sim_time}/"
        )
        makedirs(path_to_probes_dir)

        input = """# Time       p0      p1      p2      p3      p4 
            0.335       1.0     2.0     3.0     4.0     5.0     
        """
        with open(path.join(path_to_probes_dir, "p"), "w") as file:
            file.write(input)

        env = self.make_env()
        env.n_probes = n_probes
        env.latest_available_sim_time = latest_available_sim_time
        env._openfoam_solver_path = getcwd()

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        output = env._probes_to_observation()

        assert np.array_equal(output, expected)

    def test_get_action(self, testdir, mock_adapter, mocker):
        mocker.patch(
            "closed_loop_AFC.jet_control_cylinder.environment.get_interface_patches",
            return_value=["jet1", "jet2"],
        )

        sim_engine = path.join(TestJetCylinder2D.gymprecice_tutorials_dir,
                               "closed_loop_AFC/jet_control_cylinder/physics-simulation-engine")
        solver_names = ENVIRONMENT_CONFIG["solvers"]["name"]
        solver_dir = [path.join(sim_engine, solver) for solver in solver_names][0]
        print(solver_dir)
        makedirs(f"{getcwd()}/{solver_names[0]}/constant", exist_ok=True)
        system(f"cp -r {solver_dir}/constant {getcwd()}/{solver_names[0]}")

        input = np.array([-1.0, 1.0])

        env = self.make_env()
        output0 = env._action_to_patch_field(input[0])
        output1 = env._action_to_patch_field(input[1])

        assert np.array_equal(output0, np.negative(output1))

    def test_get_reward(self, testdir, patch_env_helpers, mock_adapter):
        latest_available_sim_time = 0.335
        reward_average_time_window = 1
        n_forces = 3

        path_to_forces_dir_0 = path.join(
            getcwd(), f"postProcessing/forceCoeffs/0/"
        )
        makedirs(path_to_forces_dir_0)

        path_to_forces_dir_1 = path.join(
            getcwd(), f"postProcessing/forceCoeffs/{latest_available_sim_time}/"
        )
        makedirs(path_to_forces_dir_1)

        input = """# Time    Cd   Cs   Cl 
            0.335  1.0  0    2.0     
        """
        with open(path.join(path_to_forces_dir_0, "coefficient.dat"), "w") as file:
            file.write(input)
        with open(path.join(path_to_forces_dir_1, "coefficient.dat"), "w") as file:
            file.write(input)

        env = self.make_env()
        env.n_forces = n_forces
        env.latest_available_sim_time = latest_available_sim_time
        env.reward_average_time_window = reward_average_time_window
        env._openfoam_solver_path = getcwd()

        expected = 3.205 - 1 - 0.2 * 2
        output = env._forces_to_reward()

        assert math.isclose(output, expected)
