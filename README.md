[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/gymprecice/gymprecice/blob/master/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-Commit Check](https://github.com/gymprecice/tutorials/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/gymprecice/tutorials/actions/workflows/pre-commit.yml)
## Gym-preCICE Tutorials

This repository contains ready-to-run tutorial cases that use the Gym-preCICE adapter. These tutorials can be used as a foundation for creating similar control cases.

To define new control cases, you need to follow a simple file structure with three key components:
```
new-control-problem
├── <controller-script-name>.py
├── environment.py
└── physics-simulation-engine
    ├── gymprecice-config.json
    ├── precice-config.json
    ├── solver-1
    ├── solver-2
    ├── ...
    └── solver-n
```
- `physics-simulation-engine`: is a directory containing PDE-based solver case(s), `gymprecice-config.json` file for the adapter configuration, and `precice-config.xml` file for configuring the preCICE coupling library.
- `environment.py`: is a Python script defining a class inherited from Gym-preCICE adapter to expose the underlying behaviour of the physics-simulation-engine to the controller.
- `<controller-script-name>.py`: is a Python script defining the controller algorithm that interacts with the environment. This may, for instance, be the Proximal Policy Optimisation (PPO) algorithm, the Soft-Actor-Critic (SAC) algorithm, or a simple sinusoidal signal control.

To run the control case, you need to switch to the root directory of the control case, here, `new-control-problem`, and run
 ```
 python3 -u <controller-script-name>.py
 ```
By default, the output will be saved in a directory called `gymprecice-run` that is located in the root directory of the control case. However, it is possible to specify a different path for the result directory via `gymprecice-config.json` file.

Please refer to the tutorial cases and extract the relevant sections that you require for your new control cases.


## Run a tutorial
To begin running the tutorial cases, it is necessary to have [gymprecice](https://github.com/gymprecice/gymprecice) installed beforehand.

To make sure you can successfully run the tutorials, you need to install some example-specific requirements:

- The tutorials within `closed-loop` directory rely on `OpenFOAM` CFD solvers and `OpenFOAM-preCICE adapter`. Please follow the instructions [here](https://precice.org/adapter-openfoam-overview.html) to install these dependencies.

- The tutorials within `open-loop` directory, in addition to OpenFOAM CFD solvers and OpenFOAM-preCICE adapter, rely on `deal.II` solid solvers and `deal.II-preCICE adapter`. Please follow the instructions [here](https://precice.org/adapter-dealii-overview.html) to install these dependencies.

Please check out the [Quickstart](https://github.com/gymprecice/gymprecice-tutorials/blob/master/quickstart/quickstart.ipynb) to follow running a control case step by step.

## Further instructions
The tutorials and [gymprecice](https://github.com/gymprecice/gymprecice) were tested on specific version of `preCICE`, `OpenFOAM` and `OpenFOAM-preCICE adapter` on Ubuntu 20.04.6 LTS

- `preCICE` was installed using
```
wget https://github.com/precice/precice/releases/download/v2.5.0/libprecice2_2.5.0_focal.deb
sudo apt install ./libprecice2_2.5.0_focal.deb
```
- `OpenFOAM` was installed using
```
curl https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install openfoam2112-default
```
followed by adding `source /usr/lib/openfoam/openfoam2112/etc/bashrc` to the `.bashrc` file or `.zshrc`
- `OpenFOAM-preCICE adapter` was installed locally (without sudo) using
```
wget https://github.com/precice/openfoam-adapter/releases/download/v1.1.0/openfoam-adapter_v1.1.0_OpenFOAMv1812-v2112-newer.tar.gz
tar -xzvf openfoam-adapter_v1.1.0_OpenFOAMv1812-v2112-newer.tar.gz
cd openfoam-adapter_v1.1.0_OpenFOAMv1812-v2112-newer
./Allwmake
cd ..
```

## Citing Us

If you use Gym-preCICE, please cite the following paper:

```
@misc{shams2023gymprecice,
      title={Gym-preCICE: Reinforcement learning environments for active flow control},
      author={Shams, Mosayeb and Elsheikh, Ahmed H},
      journal={SoftwareX},
      volume={23},
      pages={101446},
      year={2023},
      issn={2352-7110},
      doi={https://doi.org/10.1016/j.softx.2023.101446},
      eprint={https://arxiv.org/abs/2305.02033},
}
```


## Contributing

See the contributing guidelines [CONTRIBUTING.md](https://github.com/gymprecice/tutorials/blob/main/CONTRIBUTING.md)
for information on submitting issues and pull requests.


## The Team

Gym-preCICE and its tutorials are primarily developed and maintained by:
- Mosayeb Shams (@mosayebshams) - Lead Developer (Heriot-Watt University)
- Ahmed H. Elsheikh(@ahmed-h-elsheikh) - Co-developer and Supervisor (Heriot-Watt University)


## Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council grants number EP/V048899/1 and EP/Y006143/1.


## License

Gym-preCICE and its tutorials are [MIT licensed](https://github.com/gymprecice/tutorials/blob/main/LICENSE).
