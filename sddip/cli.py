from typing import Callable, List
import argparse

from .operators import classical_runner, dynamic_runner, extensive_runner
from .scripts import (
    clear_result_directories,
    create_scenarios,
    create_supplementary,
)


def main(argv: List[str]):
    args = _parse_arguments(argv)
    run_func = _get_run_func(args)
    run_func()


def _parse_arguments(argv: List[str]) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = _create_argument_parser()
    args = parser.parse_args(argv[1:])
    return args


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the command line interface."""

    parser = argparse.ArgumentParser(description="Dynamic SDDIP")

    parser.add_argument(
        "--classical", action="store_true", help="Run classical SDDIP"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Run dynamic SDDIP"
    )
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Run script to solve the extensive model",
    )
    parser.add_argument(
        "--scenarios", action="store_true", help="Create scenarios"
    )
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help="Run script to create supplementary data",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean result directories"
    )

    return parser


def _get_run_func(args: argparse.Namespace) -> Callable:
    if args.classical:
        return classical_runner.main
    elif args.dynamic:
        return dynamic_runner.main
    elif args.extensive:
        return extensive_runner.main
    elif args.scenarios:
        return create_scenarios.main
    elif args.supplementary:
        return create_supplementary.main
    elif args.clean:
        return clear_result_directories.main
    else:
        raise ValueError("No such execution mode.")
