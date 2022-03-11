import os
import numpy as np
import pandas as pd
import gurobipy as gp
from scipy import stats, linalg
from time import time

from sddip.ucmodel import ModelBuilder

from sddip import (
    storage,
    utils,
    logger,
    dualsolver,
    ucmodel,
    parameters,
    scenarios,
)
from sddip.constants import ResultKeys


class SddipAlgorithm:
    def __init__(
        self, test_case: str, log_dir: str, method: str = "bm", cut_mode: str = "l",
    ):
        # Logger
        self.runtime_logger = logger.RuntimeLogger(log_dir)

        # Problem specific parameters
        self.problem_params = parameters.Parameters(test_case)

        # Algorithm paramters
        self.max_n_samples = min(30, self.problem_params.n_scenarios)
        self.n_samples = self.max_n_samples
        self.n_samples_leap = 0
        self.n_binaries = 10
        self.error_threshold = 10 ** (-1)
        self.max_n_binaries = 10
        # Relative change in lower bound
        self.refinement_tolerance = 10 ** (-2)
        self.big_m = 10 ** 6
        self.sos = False

        # Helper objects
        self.binarizer = utils.Binarizer()
        self.sc_sampler = scenarios.ScenarioSampler(
            self.problem_params.n_stages,
            self.problem_params.n_realizations_per_stage[1],
        )

        ds_max_iterations = 500

        if method == "sg":
            self.dual_solver = dualsolver.SubgradientMethod(
                ds_max_iterations, 10 ** -3, log_dir
            )
        elif method == "bm":
            self.dual_solver = dualsolver.BundleMethod(
                ds_max_iterations, 10 ** -1, log_dir
            )
        else:
            raise ValueError(f"Method '{method}' does not exist.")

        self.init_cut_mode = cut_mode
        self.cut_mode = self.init_cut_mode

        self.dual_solver.log_flag = False

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
        self.bc_storage = storage.ResultStorage(
            ResultKeys.benders_cut_keys, "benders_cuts"
        )
        self.dual_solver_storage = storage.ResultStorage(
            ResultKeys.dual_solver_keys, "dual_solver"
        )

        self.bound_storage = storage.ResultStorage(ResultKeys.bound_keys, "bounds")

        # Initialization
        # self.init_binary_multipliers(self.init_n_binaries)

    def init_binary_multipliers(self, n_binaries: int):
        self.y_bin_multipliers = [
            self.binarizer.calc_binary_multipliers_from_n_binaries(ub, n_binaries)
            for ub in self.problem_params.pg_max
        ]
        self.soc_bin_multipliers = [
            self.binarizer.calc_binary_multipliers_from_n_binaries(ub, n_binaries)
            for ub in self.problem_params.soc_max
        ]

    def run(self, n_iterations: int):
        print("#### SDDiP-Algorithm started ####")
        self.runtime_logger.start()
        self.dual_solver.runtime_logger.start()
        lower_bounds = []
        for i in range(n_iterations):
            print(f"Iteration {i+1}")

            ########################################
            # Sampling
            ########################################
            sampling_start_time = time()
            n_samples = self.n_samples
            samples = self.sc_sampler.generate_samples(n_samples)
            print(f"Samples: {samples}")
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
            v_upper_l, v_upper_r = self.statistical_upper_bound(v_opt_k, n_samples)
            print("Statistical upper bound: {} ".format(v_upper_l))
            self.runtime_logger.log_task_end(
                f"upper_bound_i{i+1}", upper_bound_start_time
            )

            ########################################
            # Binary approximation refinement
            ########################################
            refinement_start_time = time()
            self.binary_approximation_refinement(i, lower_bounds)
            self.runtime_logger.log_task_end(
                f"binary_approximation_refinement_i{i+1}", refinement_start_time
            )

            ########################################
            # Backward pass
            ########################################
            backward_pass_start_time = time()
            if self.cut_mode == "l":
                self.backward_pass(i + 1, samples)
            elif self.cut_mode in ["b", "sb"]:
                self.backward_benders(i + 1, samples)
            self.runtime_logger.log_task_end(
                f"backward_pass_i{i+1}", backward_pass_start_time
            )

            ########################################
            # Lower bound
            ########################################
            lower_bound_start_time = time()
            v_lower = self.lower_bound(i + 1)
            lower_bounds.append(v_lower)
            print("Lower bound: {} ".format(v_lower))
            self.runtime_logger.log_task_end(
                f"lower_bound_i{i+1}", lower_bound_start_time
            )

            bound_dict = self.bound_storage.create_empty_result_dict()
            bound_dict[ResultKeys.lb_key] = v_lower
            bound_dict[ResultKeys.ub_l_key] = v_upper_l
            bound_dict[ResultKeys.ub_r_key] = v_upper_r
            self.bound_storage.add_result(i, 0, 0, bound_dict)

            # Increase number of samples
            if self.n_samples < self.max_n_samples:
                self.n_samples += self.n_samples_leap

        self.runtime_logger.log_experiment_end()
        self.dual_solver.runtime_logger.log_experiment_end()

        ########################################
        # Final upper bound
        ########################################
        n_samples = 30
        samples = self.sc_sampler.generate_samples(n_samples)
        v_opt_k = self.forward_pass(n_iterations, samples)
        v_upper_l, v_upper_r = self.statistical_upper_bound(v_opt_k, n_samples)

        bound_dict = self.bound_storage.create_empty_result_dict()
        bound_dict[ResultKeys.lb_key] = 0
        bound_dict[ResultKeys.ub_l_key] = v_upper_l
        bound_dict[ResultKeys.ub_r_key] = v_upper_r
        self.bound_storage.add_result(n_iterations, 0, 0, bound_dict)

        print("#### SDDiP-Algorithm finished ####")

    def forward_pass(self, iteration: int, samples: list) -> list:
        i = iteration
        n_samples = len(samples)
        v_opt_k = []

        for k in range(n_samples):
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point

            v_opt_k.append(0)
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                uc_fw = ucmodel.ForwardModelBuilder(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                )

                uc_fw: ucmodel.ForwardModelBuilder = self.add_problem_constraints(
                    uc_fw, t, n, i
                )
                # print(f"x: {x_trial_point}")
                # print(f"y: {y_trial_point}")
                # print(f"xbs: {x_bs_trial_point}")
                # print(f"soc: {soc_trial_point}")
                uc_fw.add_copy_constraints(
                    x_trial_point, y_trial_point, x_bs_trial_point, soc_trial_point,
                )

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                # if i == 5:
                #     print(f"{t}, {n}")
                #     uc_fw.enable_output()
                #     uc_fw.model.printAttr("X")

                x_kt = [x_g.x for x_g in uc_fw.x]
                y_kt = [y_g.x for y_g in uc_fw.y]
                x_bs_kt = [[x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs]
                soc_kt = [soc_s.x for soc_s in uc_fw.soc]

                # if t == 0:
                # print(f"{t}, {n}")
                # yc = [c.x for c in uc_fw.ys_c]
                # ydc = [dc.x for dc in uc_fw.ys_dc]
                #     print(f"soc-trial: {soc_trial_point}")
                #     print(f"soc: {soc_kt}")
                # print(f"yc: {yc}")
                # print(f"ydc: {ydc}")
                #     print(f"ysp: {uc_fw.ys_p.x}")
                #     print(f"ysn:{uc_fw.ys_n.x}")

                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt
                # print(f"Forward {t},{n}: {uc_fw.ys_p.x}")
                # New trial point
                # TODO trial point contains x and y
                x_trial_point = x_kt
                y_trial_point = y_kt
                x_bs_trial_point = [
                    [x_trial_point[g]] + x_bs_kt[g][:-1]
                    for g in range(self.problem_params.n_gens)
                ]
                soc_trial_point = soc_kt

                ps_dict = self.ps_storage.create_empty_result_dict()
                ps_dict[ResultKeys.x_key] = x_kt
                ps_dict[ResultKeys.y_key] = y_kt
                ps_dict[ResultKeys.x_bs_key] = x_bs_trial_point
                ps_dict[ResultKeys.soc_key] = soc_kt
                ps_dict[ResultKeys.v_key] = v_value_function

                self.ps_storage.add_result(i, k, t, ps_dict)

        return v_opt_k

    def statistical_upper_bound(self, v_opt_k: list, n_samples: int) -> float:
        v_mean = np.mean(v_opt_k)
        v_std = np.std(v_opt_k)
        alpha = 0.05

        v_upper_l = v_mean + stats.norm.ppf(alpha / 2) * v_std / np.sqrt(n_samples)
        v_uppper_r = v_mean - stats.norm.ppf(alpha / 2) * v_std / np.sqrt(n_samples)

        return v_upper_l, v_uppper_r

    def binary_approximation_refinement(self, iteration: int, lower_bounds: list):
        # Check if forward pass solution i equals that in i-1
        upper_bounds = self.problem_params.pg_max + self.problem_params.soc_max

        # errors_below_threshold = all(
        #     v <= self.error_threshold for v in continuous_variables_approx_error
        # )

        # Check if refinment condition holds
        refinement_condition = False
        if not (iteration <= 1 or self.n_binaries >= self.max_n_binaries):
            delta = abs((lower_bounds[-1] - lower_bounds[-2]) / lower_bounds[-2])
            refinement_condition = delta <= self.refinement_tolerance

        if refinement_condition:
            if self.cut_mode == "l":
                print("Refinement performed.")
                self.n_binaries += 1
            self.cut_mode = "l"

        continuous_variables_precision = [
            self.binarizer.calc_precision_from_n_binaries(ub, self.n_binaries)
            for ub in upper_bounds
        ]
        continuous_variables_approx_error = [
            self.binarizer.calc_max_abs_error(prec)
            for prec in continuous_variables_precision
        ]
        print(f"Approximation errors: {continuous_variables_approx_error}")

    def backward_pass(self, iteration: int, samples: list):
        i = iteration
        n_samples = len(samples)

        for t in reversed(range(1, self.problem_params.n_stages)):
            for k in range(n_samples):
                n_realizations = self.problem_params.n_realizations_per_stage[t]
                ds_dict = self.ds_storage.create_empty_result_dict()
                cc_dict = self.cc_storage.create_empty_result_dict()
                dual_solver_dict = self.dual_solver_storage.create_empty_result_dict()

                for n in range(n_realizations):
                    # Get mixed_integer trial points
                    y_float_vars = self.ps_storage.get_result(i - 1, k, t - 1)[
                        ResultKeys.y_key
                    ]
                    x_binary_trial_point = self.ps_storage.get_result(i - 1, k, t - 1)[
                        ResultKeys.x_key
                    ]
                    x_bs_binary_trial_point = self.ps_storage.get_result(
                        i - 1, k, t - 1
                    )[ResultKeys.x_bs_key]
                    soc_float_vars = self.ps_storage.get_result(i - 1, k, t - 1)[
                        ResultKeys.soc_key
                    ]

                    # Binarize trial points
                    y_bin_vars = []
                    y_bin_multipliers = []
                    for j in range(len(y_float_vars)):
                        (
                            new_y_vars,
                            new_y_multipliers,
                        ) = self.binarizer.binary_expansion_from_n_binaries(
                            y_float_vars[j],
                            self.problem_params.pg_max[j],
                            self.n_binaries,
                        )
                        y_bin_vars += new_y_vars
                        y_bin_multipliers.append(new_y_multipliers)

                    soc_bin_vars = []
                    soc_bin_multipliers = []
                    for j in range(len(soc_float_vars)):
                        (
                            new_soc_vars,
                            new_soc_multipliers,
                        ) = self.binarizer.binary_expansion_from_n_binaries(
                            soc_float_vars[j],
                            self.problem_params.soc_max[j],
                            self.n_binaries,
                        )
                        soc_bin_vars += new_soc_vars
                        soc_bin_multipliers.append(new_soc_multipliers)

                    # Binarized trial points
                    y_binary_trial_point = y_bin_vars
                    y_binary_trial_multipliers = linalg.block_diag(*y_bin_multipliers)
                    soc_binary_trial_point = soc_bin_vars
                    soc_binary_trial_multipliers = linalg.block_diag(
                        *soc_bin_multipliers
                    )

                    # Build backward model
                    uc_bw = ucmodel.BackwardModelBuilder(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                    )

                    uc_bw.add_relaxation(
                        x_binary_trial_point,
                        y_binary_trial_point,
                        x_bs_binary_trial_point,
                        soc_binary_trial_point,
                    )

                    uc_bw: ucmodel.BackwardModelBuilder = self.add_problem_constraints(
                        uc_bw, t, n, i
                    )

                    uc_bw.add_copy_constraints(
                        y_binary_trial_multipliers, soc_binary_trial_multipliers
                    )

                    objective_terms = uc_bw.objective_terms
                    relaxed_terms = uc_bw.relaxed_terms

                    uc_bw.disable_output()

                    binary_trial_point = (
                        x_binary_trial_point
                        + y_binary_trial_point
                        + [
                            x_bs_g
                            for x_bs in x_bs_binary_trial_point
                            for x_bs_g in x_bs
                        ]
                        + soc_binary_trial_point
                    )

                    model, sg_results = self.dual_solver.solve(
                        uc_bw.model, objective_terms, relaxed_terms,
                    )
                    dual_multipliers = sg_results.multipliers.tolist()
                    dual_value = sg_results.obj_value - np.array(dual_multipliers).dot(
                        binary_trial_point
                    )

                    # Dual value and multiplier for each realization
                    ds_dict[ResultKeys.dv_key].append(dual_value)
                    ds_dict[ResultKeys.dm_key].append(dual_multipliers)

                    dual_solver_dict[ResultKeys.ds_iterations].append(
                        sg_results.n_iterations
                    )
                    dual_solver_dict[ResultKeys.ds_solver_time].append(
                        sg_results.solver_time
                    )

                self.ds_storage.add_result(i, k, t, ds_dict)
                self.dual_solver_storage.add_result(i, k, t, dual_solver_dict)

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
                cc_dict[ResultKeys.y_bm_key] = y_binary_trial_multipliers
                cc_dict[ResultKeys.soc_bm_key] = soc_binary_trial_multipliers

                self.cc_storage.add_result(i, k, t - 1, cc_dict)

    def backward_benders(self, iteration: int, samples: list):
        i = iteration
        n_samples = len(samples)
        for t in reversed(range(1, self.problem_params.n_stages)):
            for k in range(n_samples):
                n_realizations = self.problem_params.n_realizations_per_stage[t]
                bc_dict = self.bc_storage.create_empty_result_dict()
                y_trial_point = self.ps_storage.get_result(i - 1, k, t - 1)[
                    ResultKeys.y_key
                ]
                x_trial_point = self.ps_storage.get_result(i - 1, k, t - 1)[
                    ResultKeys.x_key
                ]
                x_bs_trial_point = self.ps_storage.get_result(i - 1, k, t - 1)[
                    ResultKeys.x_bs_key
                ]
                soc_trial_point = self.ps_storage.get_result(i - 1, k, t - 1)[
                    ResultKeys.soc_key
                ]

                trial_point = (
                    x_trial_point
                    + y_trial_point
                    + [val for bs in x_bs_trial_point for val in bs]
                    + soc_trial_point
                )
                print(f"TP: {trial_point}")

                dual_multipliers = []
                opt_values = []

                for n in range(n_realizations):
                    # Create forward model
                    uc_fw = ucmodel.ForwardModelBuilder(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                        lp_relax=True,
                    )

                    uc_fw: ucmodel.ForwardModelBuilder = self.add_problem_constraints(
                        uc_fw, t, n, i
                    )

                    uc_fw.add_copy_constraints(
                        x_trial_point, y_trial_point, x_bs_trial_point, soc_trial_point,
                    )

                    uc_fw.model.optimize()
                    # if i == 1 and t == 1 and k == 0 and n == 0:
                    #     uc_fw.enable_output()
                    #     uc_fw.model.display()

                    copy_constrs = [
                        uc_fw.copy_constraints_x,
                        uc_fw.copy_constraints_y,
                        uc_fw.copy_constraints_x_bs,
                        uc_fw.copy_constraints_soc,
                    ]

                    dm = []
                    for constr_set in copy_constrs:
                        dm += [
                            constr.getAttr(gp.GRB.attr.Pi)
                            for _, constr in constr_set.items()
                        ]
                    # print(dm)
                    dual_multipliers.append(dm)

                    if self.cut_mode == "b":
                        opt_values.append(uc_fw.model.getObjective().getValue())
                    elif self.cut_mode == "sb":
                        model = ucmodel.ForwardModelBuilder(
                            self.problem_params.n_buses,
                            self.problem_params.n_lines,
                            self.problem_params.n_gens,
                            self.problem_params.n_storages,
                            self.problem_params.gens_at_bus,
                            self.problem_params.storages_at_bus,
                            self.problem_params.backsight_periods,
                        )
                        model: ucmodel.ForwardModelBuilder = self.add_problem_constraints(
                            uc_fw, t, n, i
                        )

                        copy_terms = model.get_copy_terms(
                            x_trial_point,
                            y_trial_point,
                            x_bs_trial_point,
                            soc_trial_point,
                        )

                        _, dual_value = self.dual_solver.get_subgradient_and_value(
                            uc_fw.model, uc_fw.objective_terms, copy_terms, dm
                        )
                        opt_values.append(dual_value)

                # print(f"DM: {dual_multipliers}")
                # print(f"V: {opt_values}")

                opt_values = np.array(opt_values)
                dual_multipliers = np.array(dual_multipliers)
                v = np.average(opt_values)
                pi = np.average(dual_multipliers, axis=0)

                bc_dict[ResultKeys.bc_intercept_key] = v
                bc_dict[ResultKeys.bc_gradient_key] = pi.tolist()
                bc_dict[ResultKeys.bc_trial_point_key] = [t for t in trial_point]

                self.bc_storage.add_result(i, k, t - 1, bc_dict)

    def lower_bound(self, iteration: int) -> float:
        t = 0
        n = 0
        i = iteration

        x_trial_point = self.problem_params.init_x_trial_point
        y_trial_point = self.problem_params.init_y_trial_point
        x_bs_trial_point = self.problem_params.init_x_bs_trial_point
        soc_trial_point = self.problem_params.init_soc_trial_point

        # Create forward model
        uc_fw = ucmodel.ForwardModelBuilder(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
        )

        uc_fw: ucmodel.ForwardModelBuilder = self.add_problem_constraints(
            uc_fw, t, n, i
        )

        uc_fw.add_copy_constraints(
            x_trial_point, y_trial_point, x_bs_trial_point, soc_trial_point
        )

        # Solve problem
        uc_fw.disable_output()
        uc_fw.model.optimize()

        # Value of stage t objective function
        v_lower = uc_fw.model.getObjective().getValue()

        return v_lower

    def add_problem_constraints(
        self,
        model_builder: ucmodel.ModelBuilder,
        stage: int,
        realization: int,
        iteration: int,
    ) -> ucmodel.ModelBuilder:

        model_builder.add_objective(self.problem_params.cost_coeffs)

        model_builder.add_balance_constraints(
            sum(self.problem_params.p_d[stage][realization]),
            sum(self.problem_params.re[stage][realization]),
            self.problem_params.eff_dc,
        )

        model_builder.add_power_flow_constraints(
            self.problem_params.ptdf,
            self.problem_params.pl_max,
            self.problem_params.p_d[stage][realization],
            self.problem_params.re[stage][realization],
            self.problem_params.eff_dc,
        )

        model_builder.add_storage_constraints(
            self.problem_params.rc_max,
            self.problem_params.rdc_max,
            self.problem_params.soc_max,
        )

        if stage == self.problem_params.n_stages - 1:
            model_builder.add_final_soc_constraints(
                self.problem_params.init_soc_trial_point
            )
        model_builder.add_soc_transfer(self.problem_params.eff_c)

        model_builder.add_generator_constraints(
            self.problem_params.pg_min, self.problem_params.pg_max
        )

        model_builder.add_startup_shutdown_constraints()

        model_builder.add_ramp_rate_constraints(
            self.problem_params.r_up,
            self.problem_params.r_down,
            self.problem_params.r_su,
            self.problem_params.r_sd,
        )

        model_builder.add_up_down_time_constraints(
            self.problem_params.min_up_time, self.problem_params.min_down_time
        )

        model_builder.add_cut_lower_bound(self.problem_params.cut_lb[stage])

        if stage < self.problem_params.n_stages - 1 and iteration > 0:
            if self.cut_mode == "l":
                cut_coefficients = self.cc_storage.get_stage_result(stage)
                model_builder.add_cut_constraints(
                    cut_coefficients[ResultKeys.ci_key],
                    cut_coefficients[ResultKeys.cg_key],
                    cut_coefficients[ResultKeys.y_bm_key],
                    cut_coefficients[ResultKeys.soc_bm_key],
                    self.big_m,
                    self.sos,
                )
            if self.init_cut_mode in ["b", "sb"]:
                benders_coefficients = self.bc_storage.get_stage_result(stage)
                model_builder.add_benders_cuts(
                    benders_coefficients[ResultKeys.bc_intercept_key],
                    benders_coefficients[ResultKeys.bc_gradient_key],
                    benders_coefficients[ResultKeys.bc_trial_point_key],
                )

        return model_builder
