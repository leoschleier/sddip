import argparse
import datetime as dt
import logging
import os
import tomllib
from collections.abc import Callable
from pathlib import Path

from . import config
from .operators import classical_runner, dynamic_runner, extensive_runner
from .scripts import (
    clear_result_directories,
    create_scenarios,
    create_supplementary,
    gather_latest_results,
)

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> None:
    """Run the command line interface."""
    args = _parse_arguments(argv)

    no_files = args.clean or args.gather
    _init_logging(args.verbose, no_files)
    logger.info("Executing the SDDIP package.")

    run_func = _get_run_func(args)
    if run_func:
        _load_tests(args.schedule)
        breakpoint()
        run_func()
    else:
        execution_successful = _execute_aux_func(args)

        if not execution_successful:
            logger.warning(
                "Abort execution. "
                "Unknown combination of command line arguments: %s",
                args,
            )

    logger.info("Job completed")


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = _create_argument_parser()
    return parser.parse_args(argv)


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
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("--gather", action="store_true", help="Gather results")
    parser.add_argument("-t", type=int, required=False, default=None)
    parser.add_argument("-n", type=int, required=False, default=None)
    parser.add_argument("--test-case", type=str, required=False, default=None)
    parser.add_argument(
        "--setup",
        type=Path,
        required=False,
        default=Path("setup.toml"),
        help="Path to the TOML file containing the test schedule.",
    )

    return parser


def _init_logging(verbose: bool = False, no_files: bool = False) -> None:
    """Initialize the logging."""
    if not os.path.exists(config.LOGS_DIR):
        os.makedirs(config.LOGS_DIR)

    now_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = config.LOGS_DIR / f"{now_str}_logs.txt"

    log_level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if not no_files:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
        handlers=handlers,
    )


def _load_tests(path: Path) -> list[config.TestSetup]:
    """Load the list of tests to run from the schedule file."""
    with path.open("rb") as f:
        setup = tomllib.load(f)

    tests = setup.get("tests")

    if not tests:
        msg = "No test config found."
        raise Exception(msg)

    root = Path(tests.get("root", "."))

    test_configs = []
    for conf in tests["cases"]:
        test_configs.append(
            config.TestSetup(name=conf["name"], path=root / conf["path"])
        )

    return test_configs


def _get_run_func(args: argparse.Namespace) -> Callable | None:
    """Return the function to run based on the command line arguments."""
    if args.classical:
        return classical_runner.main
    if args.dynamic:
        return dynamic_runner.main
    if args.extensive:
        return extensive_runner.main
    return None


def _execute_aux_func(args: argparse.Namespace) -> bool:
    """Execute one of the auxilliary functions if correctly specified in
    the command line arguments.
    """
    execution_successful = False
    if (
        args.t is not None
        and args.n is not None
        and args.test_case is not None
    ):
        if args.scenarios:
            create_scenarios.create_scenario_data(
                args.test_case, args.t, args.n
            )
            execution_successful = True
        elif args.supplementary:
            create_supplementary.create_supplementary_data(
                args.test_case, args.t, args.n
            )
            execution_successful = True
    elif args.gather:
        gather_latest_results.main()
        execution_successful = True
    elif args.clean:
        clear_result_directories.main()
        execution_successful = True

    return execution_successful
