from sddip import algorithm, logger, storage


# Parameters
test_case = "case6ww"
n_iterations = 1
init_n_binaries = 5
init_n_samples = 1
big_m = 10 ** 3
sos = False


# Logger
log_manager = logger.LogManager()
log_dir = log_manager.create_log_dir("log")


# Execution
algo = algorithm.SddipAlgorithm(test_case, log_dir, method="bm")
algo.big_m = big_m
algo.n_binaries = init_n_binaries
if init_n_samples:
    algo.n_samples = init_n_samples
algo.sos = sos
# algo.sg_method.output_flag = True

try:
    algo.run(n_iterations)
except KeyboardInterrupt as e:
    print("Shutdown request ... exiting")
finally:
    # Manage results
    results_manager = storage.ResultsManager()
    results_dir = results_manager.create_results_dir("results")

    algo.ps_storage.export_results(results_dir)
    algo.ds_storage.export_results(results_dir)
    algo.cc_storage.export_results(results_dir)
    algo.bound_storage.export_results(results_dir)
