# sddip

Implementation of the (dynamic) stochastic dual dynamic integer programming (SDDiP) algorithm.

## (Dynamic) SDDIP Algorithm

Multistage stochastic mixed-integer linear problems (MSMILPs) are non-convex, usually of large scale, and hard to solve. This calls for decomposition approaches to keep the solution process computationally tractable.

Recently, the set of decomposition approaches has been extended by stochastic dual dynamic integer programming (SDDiP) [1]. Constructing outer approximations of the value functions using so-called Lagrangian cuts, this method can solve MSMILPs with binary state variables exactly. Performing a binary approximation of general mixed-integer state variables, SDDiP is applicable to a broader class of MSMILPs. The binarization of variables is fixed and carried out in advance. However, this requires knowledge about problem-specific parameters which may be hard to gain in practice. A recent proposal [2] is to use a more sophisticated binarization approach where the approximation is applied temporarily and refined dynamically. For this purpose, by projection, non-convex Lagrangian cuts are created that are applicable in binary and mixed-integer state space.

The present work combines SDDiP and the dynamic binarization approach in a dynamic SDDiP algorithm. This project includes an implementation of the dynamic SDDiP algorithm as well as of the classical SDDiP approach. Both algorithms are being used to solve instances of the multistage stochastic unit commitment problem.

## Setup and Execution

Before running the contents of the present project, make sure to install the dependencies in requirements.txt. This codebase has been developed with Python 3.9. The sddip package located in the project directory can be executed using a variety of command line arguments. Run `python -m sddip -h` to find more detailed information. The command `python -m sddip --dynamic` will start the dynamic SDDiP algorithm whereas the command `python -m sddip --classical` will run the original SDDiP approach.

## References

[1] Zou, J., Ahmed, S., and Sun, X. A. Stochastic dual dynamic integer programming. Mathematical Programming 175, 461–502 (2019).

[2] Füllner, C., and Rebennack, S. Non-convex Nested Benders Decomposition Mathematical Programming 196, 987–1024 (2022).
