from ..sddip import dualsolver, logger, sddipclassical, storage
from ..sddip.sddipclassical import CutModes


def main():
    # Parameters
    test_case = "case6ww"
    n_stages = 12
    n_realizations = 6

    init_n_binaries = 10
    n_iterations = 1000
    time_limit_minutes = 24 * 60

    # Number of iterations after an unchanging
    # lower bound is considered stabilized
    stop_stabilization_count = 50
    refinement_stabilization_count = 1

    # Logger
    log_manager = logger.LogManager()
    log_dir = log_manager.create_log_dir("log")

    # Dual solver
    ds_tolerance = 10 ** -2
    ds_max_iterations = 1000
    dual_solver = dualsolver.BundleMethod(
        ds_max_iterations, ds_tolerance, log_dir
    )

    # Setup
    algo = sddipclassical.Algorithm(
        test_case, n_stages, n_realizations, log_dir, dual_solver=dual_solver,
    )
    algo.n_binaries = init_n_binaries

    algo.primary_cut_mode = CutModes.STRENGTHENED_BENDERS
    algo.n_samples_primary = 3
    algo.secondary_cut_mode = CutModes.LAGRANGIAN
    algo.n_samples_secondary = 1

    algo.time_limit_minutes = time_limit_minutes
    algo.stop_stabilization_count = stop_stabilization_count
    algo.refinement_stabilization_count = refinement_stabilization_count

    # Execution
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
            if CutModes.LAGRANGIAN in algo.cut_types_added:
                algo.cc_storage.export_results(results_dir)
            if algo.cut_types_added - set([CutModes.LAGRANGIAN]):
                algo.bc_storage.export_results(results_dir)
        except ValueError:
            print("Export incomplete.")


if __name__ == "__main__":
    main()
