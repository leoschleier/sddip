"""Execute a test session."""

import logging
import random as rdm
import time
import zoneinfo
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Literal

from sddip import config
from sddip.sddip import (
    common,
    dualsolver,
    sddipclassical,
    sddipdynamic,
)

logger = logging.getLogger(__name__)

type Seed = int | float | str | bytes | bytearray | None

@dataclass
class TestSetup:
    name: str
    path: Path
    algorithm: Literal["sddip", "dsddip"]

    sddip_n_binaries: int = field(default=5)

    sddip_max_iterations: int = field(default=100)
    sddip_time_limit: int = field(default=5 * 60)

    sddip_refinment_stabilization_count: int = field(default=5)
    sddip_stop_stabilization_count: int = field(default=1000)

    sddip_no_improvement_tolerance: float = field(default=10**-6)

    sddip_primary_cut_type: str = field(default="sb")
    sddip_n_samples_primary: int = field(default=3)
    sddip_secondary_cut_type: str = field(default="l")
    sddip_n_samples_secondary: int = field(default=1)

    sddip_projection_big_m: float = field(default=10**4)

    sddip_n_samples_final_ub: int = field(default=300)

    dual_solver_stop_tolerance: float = field(default=10**-6)
    dual_solver_time_limit: int = field(default=5 * 60)
    dual_solver_max_iterations: int = field(default=5000)

    seed: Seed = field(default_factory=lambda: int(time.time()))

    @classmethod
    def from_dict(cls, d: dict[str, Any], /) -> "TestSetup":
        """Create a `TestSetup` object from a dictionary."""
        d["path"] = Path(d["path"])
        return cls(**d)


Setup = list[TestSetup]



def start(setup: Setup, seed: Seed=None) -> None:
    """Start the test session."""
    log_manager = LogManager()
    for _test_setup in setup:
        start_time_str = dt.now(
            tz=zoneinfo.ZoneInfo("Europe/Berlin")
        ).strftime("%Y%m%d%H%M%S")
        results_dir = (
            config.RESULTS_DIR / f"{start_time_str}_{_test_setup.name}"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        log_manager.set_up_logger(results_dir / "sddip.log")

        if seed is not None:
            _test_setup.seed = seed

        run(_test_setup, str(results_dir))


def run(setup: TestSetup, results_dir: str) -> None:
    """Execute a single test."""
    logger.info("Test case: %s", setup.name)
    logger.info("Seed: %s", setup.seed)
    rdm.seed(setup.seed)

    dual_solver = dualsolver.BundleMethod(
        setup.dual_solver_max_iterations,
        setup.dual_solver_stop_tolerance,
        results_dir,
        predicted_ascent="abs",
        time_limit=setup.dual_solver_time_limit,
    )

    match setup.algorithm:
        case "sddip":
            algo = sddipclassical.Algorithm(
                setup.path,
                results_dir,
                dual_solver=dual_solver,
            )
        case "dsddip":
            algo = sddipdynamic.Algorithm(
                setup.path,
                results_dir,
                dual_solver=dual_solver,
            )
            algo.big_m = setup.sddip_projection_big_m
            algo.refinement_stabilization_count = (
                setup.sddip_refinment_stabilization_count
            )
            algo.sos = False

    algo.n_binaries = setup.sddip_n_binaries

    algo.primary_cut_mode = common.CutType.from_str(
        setup.sddip_primary_cut_type
    )
    algo.n_samples_primary = setup.sddip_n_samples_primary
    algo.secondary_cut_mode = common.CutType.from_str(
        setup.sddip_secondary_cut_type
    )
    algo.n_samples_secondary = setup.sddip_n_samples_secondary

    algo.time_limit_minutes = setup.sddip_time_limit

    # Tolerance over which an increase in the lower bound is considered
    # as an improvement.
    algo.no_improvement_tolerance = setup.sddip_no_improvement_tolerance

    # Number of iterations after an non-improving lower bound is
    # considered stabilized.
    algo.stop_stabilization_count = setup.sddip_stop_stabilization_count
    algo.n_samples_final_ub = setup.sddip_n_samples_final_ub

    # Execution
    try:
        algo.run(setup.sddip_max_iterations)
    except KeyboardInterrupt:
        logger.warning("Shutdown request received. Exiting...")
        raise
    except Exception:
        logger.exception("Execution failed: %s")
    finally:
        try:
            # Manage results
            algo.bound_storage.export_results(results_dir)
            algo.ps_storage.export_results(results_dir)
            algo.ds_storage.export_results(results_dir)
            algo.dual_solver_storage.export_results(results_dir)

            if common.CutType.LAGRANGIAN in algo.cut_types_added:
                algo.cc_storage.export_results(results_dir)
            if algo.cut_types_added - {common.CutType.LAGRANGIAN}:
                algo.bc_storage.export_results(results_dir)

        except Exception:
            logger.exception("Export incomplete: %s")
            raise

    time.sleep(1)


class LogManager:
    def __init__(self) -> None:
        self._fh: logging.FileHandler | None = None

    def set_up_logger(self, path: Path) -> None:
        """Set up logger."""
        logger = logging.getLogger("sddip")
        if self._fh:
            logger.removeHandler(self._fh)
        self._fh = logging.FileHandler(path)
        logger.addHandler(self._fh)
