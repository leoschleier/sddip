import argparse
import logging
import logging.config
import tomllib
from pathlib import Path

import sddip
from sddip import session

from .operators import extensive_runner
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
    _init_logging(args.verbose)

    logger.info("Execute SDDIP module")

    aux_exec_successful = _try_execute_aux_func(args)

    if aux_exec_successful:
        logger.info("Auxiliary function executed successfully.")
    elif args.extensive:
        logger.info("Start extensive model")
        extensive_runner.main()
    else:
        session_config = _load_session(args.session)
        logger.info("Start test session")
        session.start(session_config)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = _create_argument_parser()
    return parser.parse_args(argv)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the command line interface."""
    parser = argparse.ArgumentParser(description="Dynamic SDDIP")

    parser.add_argument(
        "--session",
        type=_path,
        required=False,
        default="session.toml",
        help="Path to the TOML file containing the session config",
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

    return parser


def _path(s: str) -> Path:
    p = Path(s)
    if not p.exists():
        msg = f"Path '{p.resolve().absolute()}' does not exist."
        raise ValueError(msg)
    return Path(s)


def _init_logging(verbose: bool = False) -> None:
    """Initialize the logging."""
    if verbose:
        for l in sddip.logging.config["loggers"].values():
            l["level"] = "DEBUG"

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


def _try_execute_aux_func(args: argparse.Namespace) -> bool:
    """Try to execute an auxiliary function.

    Return value indiactes whether an auxiliary function was executed or
    not.
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
