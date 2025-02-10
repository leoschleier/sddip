# sddip

Implementation of the (dynamic) stochastic dual dynamic integer programming
(SDDiP) algorithm.

## (Dynamic) SDDIP Algorithm

Multistage stochastic mixed-integer linear problems (MSMILPs) are non-convex,
usually of large scale, and hard to solve. This calls for decomposition
approaches to keep the solution process computationally tractable.

Recently, the set of decomposition approaches has been extended by stochastic
dual dynamic integer programming (SDDiP) [1]. Constructing outer approximations
of the value functions using so-called Lagrangian cuts, this method can solve
MSMILPs with binary state variables exactly. Performing a binary approximation
of general mixed-integer state variables, SDDiP is applicable to a broader
class of MSMILPs. The binarization of variables is fixed and carried out in
advance. However, this requires knowledge about problem-specific parameters
which may be hard to gain in practice. A recent proposal [2] is to use a more
sophisticated binarization approach where the approximation is applied
temporarily and refined dynamically. For this purpose, by projection,
non-convex Lagrangian cuts are created that are applicable in binary and
mixed-integer state space.

The present work combines SDDiP and the dynamic binarization approach in a
dynamic SDDiP algorithm. This project includes an implementation of the dynamic
SDDiP algorithm as well as of the classical SDDiP approach. Both algorithms are
being used to solve instances of the multistage stochastic unit commitment
problem.

## Setup

We use [Poetry](https://github.com/python-poetry/poetry) for packaging and
dependency management. Please use `poetry install` to create a new virtual
environment and to install the dependencies (including development
requirements) specified in the `pyproject.toml` and the `poetry.lock` file.

The above will also perform an editable install of the `sddip` package.
Therefore, when activating the virtual environment with the `poetry shell`
command, you will already be able to use the `sddip` command line interface.

Run `python -m sddip -h` to get an overview of the options that the `sddip`
command line interface provides.

## Test Sessions

To start a new test session, use the following command:

```bash
python -m sddip --session <path-to-session.toml>
```

By default, the `sddip` package will attempt to load the session config from
a `session.toml` file in the current working directory. The optional
`--session` argument enables you to select an alternative session config.

In the session config you can specify a sequence of test cases that will be
executed within one session.

A seed for the random number generator used in the scenario sampling can be set
as follows:

```bash
python -m sddip --seed <my-seed>
```

## Test Creation

To create a new test case, use the following command:

```bash
python -m sddip create --test-case <my-test-dir> --test-base <test-base-dir> -t <stages> -n <realizations>
```

With `--test-case`, you can specify the path to the target directory that will
contain the files with the data defining the new test case. This will create
new directories if the specified path does not exist yet.

The optional argument `--test-base` specifies the directory which should
contain the files defining the test system. In the current setup, this
directory is usually called `raw/`.
If a test base directory is given, all files from this directory will be copied
to the target test case directory.
If not specified, the test generation
engine will try to draw the base data from the target test case directory.

Specify the number of stages with `--stages` or `-t` and the number of
realizations per stage with  `--realizations` or `-n`. These parameters will be
used for the scenario creation.

The test generation engine will also generate the supplementary data, e.g.,
the generators' minimum up- and downtime, and ramp rates. This step can also be
triggered separately from the above with the following command:

```bash
python -m sddip create --supplementary <my-test-dir>
```

## References

[1] Zou, J., Ahmed, S., and Sun, X. A. Stochastic dual dynamic integer
programming. Mathematical Programming 175, 461–502 (2019).

[2] Füllner, C., and Rebennack, S. Non-convex Nested Benders Decomposition
Mathematical Programming 196, 987–1024 (2022).
