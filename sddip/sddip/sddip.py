import os
import numpy as np
import pandas as pd
import gurobipy as gp
from scipy import stats, linalg
from time import time

import storage as storage
import utils
import dualsolver as dualsolver
import ucmodel as ucmodel
from parameters import Parameters
import config
import logger
      

class AlgoResults:
    # Result keys
    x_key = "x"
    y_key = "y"
    z_x_key = "zx"
    z_y_key = "zy"
    primal_solution_keys = [x_key, y_key, z_x_key, z_y_key]

    dv_key = "dual_value"
    dm_key = "dual_multiplier"
    dual_solution_keys = [dv_key, dm_key]

    ci_key = "intercepts"
    cg_key = "gradients"
    bm_key = "multipliers"
    cut_coefficient_keys = [ci_key, cg_key, bm_key]
        
    # Result storage
    def __init__(self):
        self.ps_storage = storage.ResultStorage(AlgoResults.primal_solution_keys)
        self.ds_storage = storage.ResultStorage(AlgoResults.dual_solution_keys)
        self.cc_storage = storage.ResultStorage(AlgoResults.cut_coefficient_keys)

    


class SddipAlgorithm(AlgoResults):

    def __init__(self, test_case:str, log_dir:str):
        super().__init__()
        self.runtime_logger = logger.RuntimeLogger(log_dir)
        self.params = Parameters(test_case)
        self.binarizer = utils.Binarizer()
        self.sg_method = dualsolver.SubgradientMethod(max_iterations=100)

    def run(self, n_iterations = 2):
        print("#### SDDiP-Algorithm started ####")
        self.runtime_logger.start()
        for i in range(n_iterations):
            ########################################
            # Sampling
            ########################################
            # TODO sampling
            sampling_start_time = time()
            samples = [[0,1], [0,0]]
            n_samples = len(samples)
            self.runtime_logger.log_task_end(f"sampling_i{i+1}", sampling_start_time)

            ########################################
            # Forward pass
            ########################################
            forward_pass_start_time = time()
            v_opt_k = self.forward_pass(i, samples)
            self.runtime_logger.log_task_end(f"forward_pass_i{i+1}", forward_pass_start_time)
            
            ########################################
            # Statistical upper bound
            ########################################
            upper_bound_start_time = time()
            v_upper = self.statistical_upper_bound(v_opt_k, n_samples)
            print("Statistical upper bound: {} ".format(v_upper))
            self.runtime_logger.log_task_end(f"upper_bound_i{i+1}", upper_bound_start_time)


            ########################################
            # Backward pass
            ########################################
            backward_pass_start_time = time()
            self.backward_pass(i, samples)
            self.runtime_logger.log_task_end(f"backward_pass_i{i+1}", backward_pass_start_time)

            ########################################
            # Lower bound
            ########################################
            lower_bound_start_time = time()
            v_lower = self.lower_bound(i)
            print("Lower bound: {} ".format(v_upper))
            self.runtime_logger.log_task_end(f"lower_bound_i{i+1}", lower_bound_start_time)

        self.runtime_logger.log_experiment_end()
        print("#### SDDiP-Algorithm finished ####")
        
    
    def forward_pass(self, iteration:int, samples:list):
        i = iteration
        n_samples = len(samples)
        v_opt_k = []

        x_trial_point = self.params.init_x_trial_point
        y_trial_point = self.params.init_y_trial_point
        
        for k in range(n_samples):
            v_opt_k.append(0)
            for t, n in zip(range(self.params.n_stages), samples[k]):

                # Create forward model
                uc_fw = ucmodel.ForwardModelBuilder(
                    self.params.n_buses, self.params.n_lines, self.params.n_gens, self.params.gens_at_bus)

                uc_fw.add_objective(self.params.cost_coeffs)

                uc_fw.add_balance_constraints(self.params.p_d[t][n])

                uc_fw.add_power_flow_constraints(self.params.ptdf, self.params.pl_max, self.params.p_d[t][n])

                uc_fw.add_generator_constraints(self.params.pg_min, self.params.pg_max)

                uc_fw.add_startup_shutdown_constraints()

                uc_fw.add_ramp_rate_constraints(self.params.rg_up_max, self.params.rg_down_max)

                uc_fw.add_copy_constraints(x_trial_point, y_trial_point)

                #TODO Lower bound
                uc_fw.add_cut_lower_bound(self.params.cut_lb[t])
                
                if i>0:
                    cut_coefficients = self.cc_storage.get_stage_result(t)
                    uc_fw.add_cut_constraints(cut_coefficients[AlgoResults.ci_key], 
                        cut_coefficients[AlgoResults.cg_key], cut_coefficients[AlgoResults.bm_key])
                
                # Solve problem
                uc_fw.suppress_output()
                uc_fw.model.optimize()
                uc_fw.model.printAttr("X")

                # Store xtik, ytik, ztik, vtik
                x_kt = [x_g.x for x_g in uc_fw.x]
                y_kt = [y_g.x for y_g in uc_fw.y]
                z_x_kt = [z_g.x for z_g in uc_fw.z_x]
                z_y_kt = [z_g.x for z_g in uc_fw.z_y]
                s_up_kt = [s_up_g.x for s_up_g in uc_fw.s_up]
                s_down_kt = [s_down_g.x for s_down_g in uc_fw.s_down]
                
                # Value of stage t objective function
                v_opt_kt = uc_fw.model.getObjective().getValue() - uc_fw.theta.x

                v_opt_k[-1] += v_opt_kt

                # New trial point
                # TODO trial point contains x and y
                x_trial_point = x_kt
                y_trial_point = y_kt

                ps_dict = self.ps_storage.create_empty_result_dict()
                ps_dict[AlgoResults.x_key] = x_kt
                ps_dict[AlgoResults.y_key] = y_kt
                ps_dict[AlgoResults.z_x_key] = z_x_kt
                ps_dict[AlgoResults.z_y_key] = z_y_kt
                
                self.ps_storage.add_result(i, k, t, ps_dict)
        
        return v_opt_k
    
    def statistical_upper_bound(self, v_opt_k:list, n_samples):
        v_mean = np.mean(v_opt_k)
        v_std = np.std(v_opt_k)
        alpha = 0.05

        v_upper = v_mean + stats.norm.ppf(alpha/2)*v_std/np.sqrt(n_samples)
        
        return v_upper

    def backward_pass(self, iteration:int, samples:list):
        i = iteration
        n_samples = len(samples)
        
        for k in range(n_samples):
            for t in reversed(range(1,self.params.n_stages)):
                n_realizations = self.params.n_nodes_per_stage[t]
                ds_dict = self.ds_storage.create_empty_result_dict()
                cc_dict = self.cc_storage.create_empty_result_dict()
                
                for n in range(n_realizations):

                    # TODO Binarization
                    bin_vars = []
                    bin_multipliers = []
                    if t>0:
                        float_vars = self.ps_storage.get_result(i,k,t-1)[AlgoResults.y_key]
                        x_binary_trial_point = self.ps_storage.get_result(i,k,t-1)[AlgoResults.x_key]
                    else:
                        #TODO Approximation needed?
                        # Might lead to active penalty
                        float_vars = np.zeros(self.params.n_gens)
                        x_binary_trial_point = np.zeros(self.params.n_gens)
                    
                    for j in range(len(float_vars)):
                        new_vars, new_multipliers = self.binarizer.binary_expansion(
                            float_vars[j], upper_bound=self.params.pg_max, precision=0.5)
                        bin_vars += new_vars
                        bin_multipliers.append(new_multipliers) 

                    # Binarized trial points
                    y_binary_trial_point = bin_vars
                    y_binary_trial_multipliers = linalg.block_diag(*bin_multipliers)

                    # Buzild backward model
                    uc_bw = ucmodel.BackwardModelBuilder(self.params.n_buses, self.params.n_lines, self.params.n_gens, 
                        self.params.gens_at_bus)

                    uc_bw.add_objective(self.params.cost_coeffs)

                    uc_bw.add_balance_constraints(self.params.p_d[t][n])

                    uc_bw.add_generator_constraints(self.params.pg_min, self.params.pg_max)

                    uc_bw.add_power_flow_constraints(self.params.ptdf, self.params.pl_max, self.params.p_d[t][n])

                    uc_bw.add_startup_shutdown_constraints()

                    uc_bw.add_ramp_rate_constraints(self.params.rg_up_max, self.params.rg_down_max)

                    uc_bw.add_relaxation(x_binary_trial_point, y_binary_trial_point)

                    uc_bw.add_copy_constraints(y_binary_trial_multipliers)

                    uc_bw.add_cut_lower_bound(self.params.cut_lb[t])
                    
                    if t < self.params.n_stages-1:
                        cut_coefficients = self.cc_storage.get_stage_result(t)
                        uc_bw.add_cut_constraints(cut_coefficients[AlgoResults.ci_key], 
                            cut_coefficients[AlgoResults.cg_key], cut_coefficients[AlgoResults.bm_key])

                    objective_terms = uc_bw.objective_terms
                    relaxed_terms = uc_bw.relaxed_terms
                    
                    # Solve problem with subgradient method
                    uc_bw.suppress_output()            
                    model, sg_results = self.sg_method.solve(uc_bw.model, objective_terms, relaxed_terms, 10000)
                    model.printAttr("X")

                    
                    # Dual value and multiplier for each realization
                    ds_dict[AlgoResults.dv_key].append(sg_results.obj_value)
                    ds_dict[AlgoResults.dm_key].append(sg_results.multipliers)
                    
                
                self.ds_storage.add_result(i, k, t, ds_dict)

                # Calculate and store cut coefficients
                probabilities = self.params.prob[t]        
                intercept = np.array(probabilities).dot(np.array(ds_dict[AlgoResults.dv_key]))
                gradient = np.array(probabilities).dot(np.array(ds_dict[AlgoResults.dm_key]))

                cc_dict[AlgoResults.ci_key] = intercept.tolist()
                cc_dict[AlgoResults.cg_key] = gradient.tolist()
                cc_dict[AlgoResults.bm_key] = y_binary_trial_multipliers

                if t > 0 : self.cc_storage.add_result(i, k, t-1, cc_dict)

    def lower_bound(self, iteration:int):
        i = iteration
        t = 0
        n = 0
        
        x_trial_point = self.params.init_x_trial_point
        y_trial_point = self.params.init_y_trial_point

        # Create forward model
        uc_fw = ucmodel.ForwardModelBuilder(self.params.n_buses, self.params.n_lines, self.params.n_gens, 
            self.params.gens_at_bus)

        uc_fw.add_objective(self.params.cost_coeffs)

        uc_fw.add_balance_constraints(self.params.p_d[t][n])

        uc_fw.add_power_flow_constraints(self.params.ptdf, self.params.pl_max, self.params.p_d[t][n])

        uc_fw.add_generator_constraints(self.params.pg_min, self.params.pg_max)

        uc_fw.add_startup_shutdown_constraints()

        uc_fw.add_ramp_rate_constraints(self.params.rg_up_max, self.params.rg_down_max)

        uc_fw.add_copy_constraints(x_trial_point, y_trial_point)

        #TODO Lower bound
        uc_fw.add_cut_lower_bound(self.params.cut_lb[t])

        if i>0:
            cut_coefficients = self.cc_storage.get_stage_result(t)
            uc_fw.add_cut_constraints(cut_coefficients[AlgoResults.ci_key], cut_coefficients[AlgoResults.cg_key], 
                cut_coefficients[AlgoResults.bm_key])

        # Solve problem
        uc_fw.suppress_output()
        uc_fw.model.optimize()
        uc_fw.model.printAttr("X")

        # Value of stage t objective function
        v_lower = uc_fw.model.getObjective().getValue()

        return v_lower


    #TODO sampling
    def sample(self, n_realizations:list):
        pass



if __name__ == '__main__':
    log_manager = logger.LogManager()
    log_dir = log_manager.create_log_dir("log")
    algorithm = SddipAlgorithm("WB2", log_dir)
    algorithm.run()
