# import sys, os

# sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "sddip"))

from sddip import logger, sddipdynamic, storage
from sddip.dualsolver import DualSolverMethods
from sddip.sddipdynamic import CutModes


def main():
    # Parameters
    test_case = "case30"
    n_stages = 6
    n_realizations = 6

    n_iterations = 100
    time_limit_minutes = 5 * 60

    # Number of iterations after an unchanging
    # lower bound is considered stabilized
    stop_stabilization_count = 50
    refinement_stabilization_count = 5

    init_n_binaries = 5

    # Gradual increase in number of samples
    n_samples_leap = 0

    # Starting cut mode
    # b: Bender's cuts
    # sb: Strengthened Benders' cuts
    # l: Lagrangian cuts
    # If starting cut mode is 'l', then it will not be changed throughout the algorithm
    init_cut_mode = CutModes.STRENGTHENED_BENDERS
    init_n_samples = 3

    # Note: sos-constraint in the cut projection cannot be used
    # in combination with Benders cuts due to the LP relaxation
    sos = False
    big_m = 10 ** 3

    # Logger
    log_manager = logger.LogManager()
    log_dir = log_manager.create_log_dir("log")

    # Execution
    algo = sddipdynamic.Algorithm(
        test_case,
        n_stages,
        n_realizations,
        log_dir,
        dual_solver_method=DualSolverMethods.BUNDLE_METHOD,
        cut_mode=init_cut_mode,
    )
    algo.big_m = big_m
    algo.n_binaries = init_n_binaries
    if init_n_samples:
        algo.n_samples = init_n_samples
    algo.n_samples_leap = n_samples_leap
    algo.sos = sos
    algo.time_limit_minutes = time_limit_minutes
    algo.stop_stabilization_count = stop_stabilization_count
    algo.refinement_stabilization_count = refinement_stabilization_count

    try:
        algo.run(n_iterations)
    except KeyboardInterrupt as e:
        print("Shutdown request ... exiting")
    finally:
        try:
            # Manage results
            results_manager = storage.ResultsManager()
            results_dir = results_manager.create_results_dir("results")
            algo.bound_storage.export_results(results_dir)
            algo.ps_storage.export_results(results_dir)
            algo.ds_storage.export_results(results_dir)
            algo.dual_solver_storage.export_results(results_dir)
            if init_cut_mode == CutModes.LAGRANGIAN:
                algo.cc_storage.export_results(results_dir)
            else:
                algo.bc_storage.export_results(results_dir)
                algo.cc_storage.export_results(results_dir)
        except ValueError:
            print("Export incomplete.")


if __name__ == "__main__":
    main()
