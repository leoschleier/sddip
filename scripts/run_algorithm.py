from sddip import algorithm, logger, storage


# Parameters
test_case = "case6ww"
n_iterations = 15
n_samples = 3
init_n_binaries = 10
big_m = 10 ** 18
sos = True


# Logger
log_manager = logger.LogManager()
log_dir = log_manager.create_log_dir("log")


# Execution
algo = algorithm.SddipAlgorithm(test_case, log_dir)
algo.big_m = big_m
algo.n_samples = n_samples
algo.n_binaries = init_n_binaries
algo.sos = sos
# algo.sg_method.output_flag = True
algo.run(n_iterations)


# Manage results
results_manager = storage.ResultsManager()
results_dir = results_manager.create_results_dir("results")

algo.ps_storage.export_results(results_dir)
algo.ds_storage.export_results(results_dir)
algo.cc_storage.export_results(results_dir)
