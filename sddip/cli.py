from typing import Callable, List

from .operators import classical_runner, dynamic_runner, extensive_runner
from .scripts import (
    clear_result_directories,
    create_scenarios,
    create_supplementary,
)

CLASSICAL_MODE = "classical"
DYNAMIC_MODE = "dynamic"
EXTENSIVE_MODE = "extensive"
SCENARIO_CREATION = "scenarios"
SUPPLEMENTARY_CREATION = "supplementary"
CLEAR_RESULTS = "clean"


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
    elif mode == SCENARIO_CREATION:
        return create_scenarios.main
    elif mode == SUPPLEMENTARY_CREATION:
        return create_supplementary.main
    elif mode == CLEAR_RESULTS:
        return clear_result_directories.main
    else:
        raise ValueError("No such mode.")

