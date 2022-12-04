import logging
from typing import Callable, List
import argparse

from .operators import classical_runner, dynamic_runner, extensive_runner
from .scripts import (
    clear_result_directories,
    create_scenarios,
    create_supplementary,
)


logger = logging.getLogger(__name__)


def main(argv: List[str]):
    """Run the command line interface."""
    args = _parse_arguments(argv)

    _init_logging()
    logger.info("Executing the SDDIP package.")

    run_func = _get_run_func(args)
    run_func()


def _parse_arguments(argv: List[str]) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = _create_argument_parser()
    args = parser.parse_args(argv)
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


def _init_logging():
    """Initialize the logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )


def _get_run_func(args: argparse.Namespace) -> Callable:
    """Return the function to run based on the command line arguments."""
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
