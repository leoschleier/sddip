import os
import numpy as np
import pandas as pd
import gurobipy as gp
from scipy import stats, linalg
from time import time

from sddip import storage, utils, logger, dualsolver, ucmodel, parameters, scenarios
from sddip.constants import ResultKeys


class SddipAlgorithm:
    def __init__(self, test_case: str, log_dir: str):
        # Logger
        self.runtime_logger = logger.RuntimeLogger(log_dir)

        # Algorithm paramters
        self.n_samples = 30
        self.n_binaries = 15
        self.big_m = 10 ** 18
        self.sos = False

        # Problem specific parameters
        self.problem_params = parameters.Parameters(test_case)

        # Helper objects
        self.binarizer = utils.Binarizer()
        self.sc_sampler = scenarios.ScenarioSampler(
            self.problem_params.n_stages, self.problem_params.n_nodes_per_stage[1]
        )
        self.sg_method = dualsolver.SubgradientMethod(
            max_iterations=100, log_dir=log_dir
        )
        self.sg_method.log_flag = False

        # Result storage
        self.ps_storage = storage.ResultStorage(
            ResultKeys.primal_solution_keys, "primal_solutions"
        )
        self.ds_storage = storage.ResultStorage(
            ResultKeys.dual_solution_keys, "dual_solutions"
        )
        self.cc_storage = storage.ResultStorage(
            ResultKeys.cut_coefficient_keys, "cut_coefficients"
        )

        # Initialization
        # self.init_binary_multipliers(self.init_n_binaries)

    def init_binary_multipliers(self, n_binaries: int):
        self.y_bin_multipliers = [
            self.binarizer.calc_binary_multipliers_from_n_binaries(ub, n_binaries)
            for ub in self.problem_params.pg_max
        ]

    def run(self, n_iterations: int):
        print("#### SDDiP-Algorithm started ####")
        self.runtime_logger.start()
        for i in range(n_iterations):
            ########################################
            # Sampling
            ########################################
            sampling_start_time = time()
            n_samples = self.n_samples
            samples = self.sc_sampler.generate_samples(n_samples)
            self.runtime_logger.log_task_end(f"sampling_i{i+1}", sampling_start_time)

            ########################################
            # Forward pass
            ########################################
            forward_pass_start_time = time()
            v_opt_k = self.forward_pass(i, samples)
            self.runtime_logger.log_task_end(
                f"forward_pass_i{i+1}", forward_pass_start_time
            )

            ########################################
            # Statistical upper bound
            ########################################
            upper_bound_start_time = time()
            v_upper = self.statistical_upper_bound(v_opt_k, n_samples)
            print("Statistical upper bound: {} ".format(v_upper))
            self.runtime_logger.log_task_end(
                f"upper_bound_i{i+1}", upper_bound_start_time
            )

            ########################################
            # Binary approximation refinement
            ########################################
            refinement_start_time = time()
            self.binary_approximation_refinement()
            self.runtime_logger.log_task_end(
                f"binary_approximation_refinement_i{i+1}", refinement_start_time
            )

            ########################################
            # Backward pass
            ########################################
            backward_pass_start_time = time()
            self.backward_pass(i, samples)
            self.runtime_logger.log_task_end(
                f"backward_pass_i{i+1}", backward_pass_start_time
            )

            ########################################
            # Lower bound
            ########################################
            lower_bound_start_time = time()
            v_lower = self.lower_bound()
            print("Lower bound: {} ".format(v_lower))
            self.runtime_logger.log_task_end(
                f"lower_bound_i{i+1}", lower_bound_start_time
            )

        self.runtime_logger.log_experiment_end()
        print("#### SDDiP-Algorithm finished ####")

    def forward_pass(self, iteration: int, samples: list) -> list:
        i = iteration
        n_samples = len(samples)
        v_opt_k = []

        x_trial_point = self.problem_params.init_x_trial_point
        y_trial_point = self.problem_params.init_y_trial_point
        x_bs_trial_point = self.problem_params.init_x_bs_trial_point

        for k in range(n_samples):
            v_opt_k.append(0)
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                uc_fw = ucmodel.ForwardModelBuilder(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.gens_at_bus,
                    self.problem_params.backsight_periods,
                )

                uc_fw: ucmodel.ForwardModelBuilder = self.add_problem_constraints(
                    uc_fw, t, n
                )

                uc_fw.add_copy_constraints(
                    x_trial_point, y_trial_point, x_bs_trial_point
                )

                if i > 0:
                    cut_coefficients = self.cc_storage.get_stage_result(t)
                    uc_fw.add_cut_constraints(
                        cut_coefficients[ResultKeys.ci_key],
                        cut_coefficients[ResultKeys.cg_key],
                        cut_coefficients[ResultKeys.bm_key],
                        self.big_m,
                        self.sos,
                    )

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                uc_fw.model.printAttr("X")

                # Store xtik, ytik, ztik, vtik
                x_kt = [x_g.x for x_g in uc_fw.x]
                y_kt = [y_g.x for y_g in uc_fw.y]
                x_bs_kt = [[x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs]
                # z_x_kt = [z_g.x for z_g in uc_fw.z_x]
                # z_y_kt = [z_g.x for z_g in uc_fw.z_y]
                # s_up_kt = [s_up_g.x for s_up_g in uc_fw.s_up]
                # s_down_kt = [s_down_g.x for s_down_g in uc_fw.s_down]

                # Value of stage t objective function
                v_opt_kt = uc_fw.model.getObjective().getValue() - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt

                # New trial point
                # TODO trial point contains x and y
                x_trial_point = x_kt
                y_trial_point = y_kt
                x_bs_trial_point = [
                    [x_trial_point[g]] + x_bs_kt[g][:-1]
                    for g in range(self.problem_params.n_gens)
                ]

                ps_dict = self.ps_storage.create_empty_result_dict()
                ps_dict[ResultKeys.x_key] = x_kt
                ps_dict[ResultKeys.y_key] = y_kt
                ps_dict[ResultKeys.x_bs_key] = x_bs_trial_point

                self.ps_storage.add_result(i, k, t, ps_dict)

        return v_opt_k

    def statistical_upper_bound(self, v_opt_k: list, n_samples: int) -> float:
        v_mean = np.mean(v_opt_k)
        v_std = np.std(v_opt_k)
        alpha = 0.05

        v_upper = v_mean + stats.norm.ppf(alpha / 2) * v_std / np.sqrt(n_samples)

        return v_upper

    def binary_approximation_refinement(self):
        # TODO refinement condition
        # Check if forward pass solution i equals that in i-1
        refinement_condition = False

        new_multipliers = []
        if refinement_condition:
            self.n_binaries += 1
            # for g in range(self.problem_params.n_gens):
            #     n_binaries = len(self.y_bin_multipliers[g]) + 1
            #     new_multipliers.append(
            #         self.binarizer.calc_binary_multipliers_from_n_binaries(
            #             self.problem_params.pg_max[g], n_binaries
            #         )
            #     )

            # self.y_bin_multipliers = new_multipliers

    def backward_pass(self, iteration: int, samples: list):
        i = iteration
        n_samples = len(samples)

        for t in reversed(range(1, self.problem_params.n_stages)):
            for k in range(n_samples):
                n_realizations = self.problem_params.n_nodes_per_stage[t]
                ds_dict = self.ds_storage.create_empty_result_dict()
                cc_dict = self.cc_storage.create_empty_result_dict()

                for n in range(n_realizations):

                    bin_vars = []
                    y_bin_multipliers = []
                    if t > 0:
                        y_float_vars = self.ps_storage.get_result(i, k, t - 1)[
                            ResultKeys.y_key
                        ]
                        x_binary_trial_point = self.ps_storage.get_result(i, k, t - 1)[
                            ResultKeys.x_key
                        ]
                        x_bs_binary_trial_point = self.ps_storage.get_result(
                            i, k, t - 1
                        )[ResultKeys.x_bs_key]
                    else:
                        # TODO Approximation needed?
                        # TODO Initial (x,y)
                        # Might lead to active penalty
                        y_float_vars = np.zeros(self.problem_params.n_gens)
                        x_binary_trial_point = np.zeros(self.problem_params.n_gens)
                        x_bs_binary_trial_point = [
                            [0] * periods
                            for periods in self.problem_params.backsight_periods
                        ]
                    for j in range(len(y_float_vars)):
                        (
                            new_vars,
                            new_multipliers,
                        ) = self.binarizer.binary_expansion_from_n_binaries(
                            y_float_vars[j],
                            self.problem_params.pg_max[j],
                            self.n_binaries,
                        )
                        bin_vars += new_vars
                        y_bin_multipliers.append(new_multipliers)

                    # Binarized trial points
                    y_binary_trial_point = bin_vars
                    y_binary_trial_multipliers = linalg.block_diag(*y_bin_multipliers)

                    # Build backward model
                    uc_bw = ucmodel.BackwardModelBuilder(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.gens_at_bus,
                        self.problem_params.backsight_periods,
                    )

                    uc_bw: ucmodel.BackwardModelBuilder = self.add_problem_constraints(
                        uc_bw, t, n
                    )

                    uc_bw.add_relaxation(
                        x_binary_trial_point,
                        y_binary_trial_point,
                        x_bs_binary_trial_point,
                    )

                    uc_bw.add_copy_constraints(y_binary_trial_multipliers)

                    if t < self.problem_params.n_stages - 1:
                        cut_coefficients = self.cc_storage.get_stage_result(t)
                        uc_bw.add_cut_constraints(
                            cut_coefficients[ResultKeys.ci_key],
                            cut_coefficients[ResultKeys.cg_key],
                            cut_coefficients[ResultKeys.bm_key],
                            self.big_m,
                            self.sos,
                        )

                    objective_terms = uc_bw.objective_terms
                    relaxed_terms = uc_bw.relaxed_terms

                    # Solve problem with subgradient method
                    uc_bw.disable_output()

                    model, sg_results = self.sg_method.solve(
                        uc_bw.model,
                        objective_terms,
                        relaxed_terms,
                        100,
                        log_id=f"{i}_{k}_{t}_{n}",
                    )

                    model.printAttr("X")

                    # Dual value and multiplier for each realization
                    binary_trial_point = (
                        x_binary_trial_point
                        + y_binary_trial_point
                        + [
                            x_bs_g
                            for x_bs in x_bs_binary_trial_point
                            for x_bs_g in x_bs
                        ]
                    )
                    dual_multipliers = sg_results.multipliers.tolist()
                    dual_value = sg_results.obj_value - np.array(dual_multipliers).dot(
                        binary_trial_point
                    )
                    ds_dict[ResultKeys.dv_key].append(dual_value)
                    ds_dict[ResultKeys.dm_key].append(dual_multipliers)

                self.ds_storage.add_result(i, k, t, ds_dict)

                # Calculate and store cut coefficients
                probabilities = self.problem_params.prob[t]
                intercept = np.array(probabilities).dot(
                    np.array(ds_dict[ResultKeys.dv_key])
                )
                gradient = np.array(probabilities).dot(
                    np.array(ds_dict[ResultKeys.dm_key])
                )

                cc_dict[ResultKeys.ci_key] = intercept.tolist()
                cc_dict[ResultKeys.cg_key] = gradient.tolist()
                cc_dict[ResultKeys.bm_key] = y_binary_trial_multipliers

                if t > 0:
                    self.cc_storage.add_result(i, k, t - 1, cc_dict)

    def lower_bound(self) -> float:
        t = 0
        n = 0

        x_trial_point = self.problem_params.init_x_trial_point
        y_trial_point = self.problem_params.init_y_trial_point
        x_bs_trial_point = self.problem_params.init_x_bs_trial_point

        # Create forward model
        uc_fw = ucmodel.ForwardModelBuilder(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.gens_at_bus,
            self.problem_params.backsight_periods,
        )

        uc_fw: ucmodel.ForwardModelBuilder = self.add_problem_constraints(uc_fw, t, n)

        uc_fw.add_copy_constraints(x_trial_point, y_trial_point, x_bs_trial_point)

        cut_coefficients = self.cc_storage.get_stage_result(t)
        uc_fw.add_cut_constraints(
            cut_coefficients[ResultKeys.ci_key],
            cut_coefficients[ResultKeys.cg_key],
            cut_coefficients[ResultKeys.bm_key],
            self.big_m,
            self.sos,
        )

        # Solve problem
        uc_fw.disable_output()
        uc_fw.model.optimize()
        uc_fw.model.printAttr("X")

        # Value of stage t objective function
        v_lower = uc_fw.model.getObjective().getValue()

        return v_lower

    def add_problem_constraints(
        self, model_builder: ucmodel.ModelBuilder, stage: int, realization: int
    ) -> ucmodel.ModelBuilder:
        model_builder.add_objective(self.problem_params.cost_coeffs)

        model_builder.add_balance_constraints(
            sum(self.problem_params.p_d[stage][realization])
        )

        model_builder.add_power_flow_constraints(
            self.problem_params.ptdf,
            self.problem_params.pl_max,
            self.problem_params.p_d[stage][realization],
        )

        model_builder.add_generator_constraints(
            self.problem_params.pg_min, self.problem_params.pg_max
        )

        model_builder.add_startup_shutdown_constraints()

        model_builder.add_ramp_rate_constraints(
            self.problem_params.rg_up_max, self.problem_params.rg_down_max
        )

        model_builder.add_up_down_time_constraints(
            self.problem_params.min_up_times, self.problem_params.min_down_times
        )

        model_builder.add_cut_lower_bound(self.problem_params.cut_lb[stage])

        return model_builder
