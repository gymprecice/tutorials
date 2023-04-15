[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/gymprecice/gymprecice/blob/master/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
## Gym-preCICE Tutorials

This repository contains ready-to-run tutorial cases for the Gym-preCICE adapter. These tutorials can be used as a foundation for creating similar control cases.

To define new control cases, you need to follow a simple file structure with three key components:
```
new-control-problem
├── <controller-script-name>.py
├── envrionment.py
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

- The tutorials within `closed-loop` directory rely on `OpenFOAM` CFD solvers and `OpenFOAM-preCICE adapter`. Please follow the insructions [here](https://precice.org/adapter-openfoam-overview.html) to install these dependencies.

- The tutorials within `open-loop` directory, in addition to OpenFOAM CFD solvers and OpenFOAM-preCICE adapter, rely on `deal.II` solid solvers and `deal.II-preCICE adapter`. Please follow the insructions [here](https://precice.org/adapter-dealii-overview.html) to install these dependencies.

Please check out the [Quickstart](https://github.com/gymprecice/gymprecice-tutorials/blob/master/quickstart/quickstart.ipynb) to follow running a control case step by step.
## Contributions
gymprecice-tutorials is currently developed and maintained by: 

- Mosayeb Shams (@mosayebshams) - Lead developer
- Ahmed H. Elsheikh(@ahmed-h-elsheikh) - Co developer and Supervisor 


## Citation
If you use this repository, please consider citing its technical paper:

```
@misc{,
  Author = {Mosayeb Shams and Ahmed H. Elsheikh},
  Title = {Gym-preCICE: Coupling Reinforcement Learning Algorithms with External Physics-Based Solvers for Active Flow Control},
  Year = {2023},
  Eprint = {arXiv:},
}
```

## License
gymprecice-tutorials is MIT-licensed; Please refer to the [LICENSE](https://github.com/gymprecice/blob/main/LICENSE) file for more information.
