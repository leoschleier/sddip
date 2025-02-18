import argparse
import contextlib
import logging
import logging.config
import tomllib
from pathlib import Path

import sddip
from sddip import session

from .operators import extensive_runner
from .scripts import (
    clear_result_directories,
    copy_raw,
    create_scenarios,
    create_supplementary,
    gather_latest_results,
)

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> None:
    """Run the command line interface."""
    args = _parse_arguments(argv)

    match args.command:
        case "create":
            _init_logging(args.verbose, nofile=True)
            _create(args)
        case "sweep":
            _init_logging(args.verbose, nofile=True)
            _sweep(args)
        case _:
            _init_logging(args.verbose)
            logger.info("Execute SDDIP module")
            if args.extensive:
                logger.info("Start extensive model")
                extensive_runner.main()
            else:
                session_config = _load_session(args.session)
                logger.info("Start test session")
                session.start(setup=session_config, seed=args.seed)
            logger.info("Execution completed")


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = _create_argument_parser()
    return parser.parse_args(argv)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the command line interface."""
    parser = argparse.ArgumentParser(description="Dynamic SDDIP")

    parser.add_argument(
        "--session",
        type=Path,
        default="sessions/demo.toml",
        help="Path to the TOML file containing the session config",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Run script to solve the extensive model",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(
        title="Auxiliary functions", dest="command"
    )

    create_sp = subparsers.add_parser(
        "create", help="Create data for a new test case"
    )
    create_sp.add_argument(
        "--test-case",
        type=Path,
        required=True,
        default=None,
        help="Directory for test case data",
    )
    create_sp.add_argument(
        "--test-base",
        type=Path,
        default=None,
        help="Directory with basic data for a test case",
    )
    create_sp.add_argument(
        "--stages", "-t", type=int, default=None, help="Number of stages"
    )
    create_sp.add_argument(
        "--realizations",
        "-n",
        type=int,
        default=None,
        help="Number of realizations per stage",
    )
    create_sp.add_argument(
        "--supplementary",
        action="store_true",
        required=False,
        help="Create supplementary data only",
    )

    sweep_sp = subparsers.add_parser("sweep", help="Sweep results")
    sweep_sp.add_argument(
        "--gather", action="store_true", help="Gather results"
    )
    sweep_sp.add_argument(
        "--clean", action="store_true", help="Clean result directories"
    )

    return parser


def _init_logging(verbose: bool = False, nofile: bool = False) -> None:
    """Initialize the logging."""
    loggers = sddip.logging.config["loggers"]
    if verbose:
        for l in loggers.values():
            l["level"] = "DEBUG"
    if nofile:
        sddip.logging.config["handlers"].pop("file")
        for l in loggers.values():
            handlers = l["handlers"]
            with contextlib.suppress(ValueError):
                handlers.remove("file")
    else:
        sddip.logging.create_logging_dir()

    logging.config.dictConfig(sddip.logging.config)


def _load_session(path: Path) -> session.Setup:
    """Load the list of tests to run from the schedule file."""
    with path.open("rb") as f:
        setup = tomllib.load(f)

    tests = setup.get("tests")

    if not tests:
        msg = "No test config found."
        raise Exception(msg)

    return [session.TestSetup.from_dict(case) for case in tests["cases"]]


def _create(args: argparse.Namespace) -> None:
    """Create data for a new test case."""
    if args.supplementary:
        create_supplementary.create_supplementary_data(
            source_dir=args.test_base or args.test_case,
            target_dir=args.test_case,
        )
    elif args.stages and args.realizations:
        create_scenarios.create_scenario_data(
            args.stages, args.realizations, args.test_case, args.test_base
        )
        create_supplementary.create_supplementary_data(
            source_dir=args.test_base or args.test_case,
            target_dir=args.test_case,
        )
    else:
        msg = "Invalid arguments for number of stages and realizations."
        raise ValueError(msg)

    if args.test_base:
        copy_raw.move_files(args.test_base, args.test_case)


def _sweep(args: argparse.Namespace) -> None:
    """Sweep results."""
    if args.gather:
        gather_latest_results.main()
    elif args.clean:
        clear_result_directories.main()
    else:
        msg = "Invalid arguments for sweep command."
        raise ValueError(msg)
