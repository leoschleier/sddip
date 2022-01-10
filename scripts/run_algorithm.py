from sddip import algorithm, logger, storage


# Parameters
test_case = "LMBM3"
n_iterations = 3
n_samples = 30
init_precision = 0.5
big_m = 10 ** 18


# Logger
log_manager = logger.LogManager()
log_dir = log_manager.create_log_dir("log")


# Execution
algo = algorithm.SddipAlgorithm(test_case, log_dir)
algo.big_m = big_m
algo.n_samples = n_samples
algo.init_precision = init_precision
algo.run(n_iterations)


# Manage results
results_manager = storage.ResultsManager()
results_dir = results_manager.create_results_dir("results")

algo.ps_storage.export_results(results_dir)
algo.ds_storage.export_results(results_dir)
algo.cc_storage.export_results(results_dir)