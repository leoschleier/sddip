from typing import Callable, List

from .operators import classical_runner, dynamic_runner, extensive_runner


CLASSICAL_MODE = "classical"
DYNAMIC_MODE = "dynamic"
EXTENSIVE_MODE = "extensive"


def main(argv: List[str]):
    mode = argv[0]
    run_func = _get_run_func(mode)
    run_func()


def _get_run_func(mode: str) -> Callable:
    if mode == CLASSICAL_MODE:
        return classical_runner.main
    elif mode == DYNAMIC_MODE:
        return dynamic_runner.main
    elif mode == EXTENSIVE_MODE:
        return extensive_runner.main
    else:
        raise ValueError("No such mode.")

