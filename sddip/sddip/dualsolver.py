import gurobipy as gp
import numpy as np
from time import time
import os
from sddip import logger


class SubgradientMethod:
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 10 ** (-3),
        initial_lower_bound: float = 0,
        log_dir: str = None,
    ) -> None:
        self.inital_lower_bound = initial_lower_bound
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.results = SolverResults()

        self.output_flag = False
        self.output_verbose = False
        self.log_flag = False

        log_manager = logger.LogManager()
        runtime_log_dir = log_manager.create_log_dir("sg_log")
        self.runtime_logger = logger.RuntimeLogger(runtime_log_dir)
        self.n_calls = 0

        self.log_dir = log_dir

        # Step size parameters
        self.const_step_size = 1
        self.const_step_length = 1
        self.step_size_parameter = 2
        self.no_improvement_limit = 10

    def solve(
        self,
        model: gp.Model,
        objective_terms,
        relaxed_terms,
        upper_bound: float = None,
        log_id: str = None,
    ) -> gp.Model:

        self.n_calls += 1
        subgradient_start_time = time()

        model.setParam("OutputFlag", 0)
        if self.log_flag:
            current_log_dir = self.create_subgradient_log_dir(log_id)
            gurobi_logger = logger.GurobiLogger(current_log_dir)

        self.results = SolverResults()

        gradient_len = len(relaxed_terms)
        dual_multipliers = np.zeros(gradient_len)

        best_lower_bound = self.inital_lower_bound
        best_multipliers = dual_multipliers

        no_improvement_counter = 0

        step_size_parameter = self.step_size_parameter
        step_size = None

        tolerance_reached = False

        self.print_info("Subgradient Method started")

        for j in range(self.max_iterations):
            # Update model
            total_objective = objective_terms + gp.quicksum(
                relaxed_terms[i] * dual_multipliers[i] for i in range(gradient_len)
            )
            model.setObjective(total_objective)
            model.update()

            # Optimization
            model.optimize()

            if self.log_flag:
                gurobi_logger.log_model(
                    model, str(j).zfill(len(str(self.max_iterations)))
                )

            # Optimal value
            opt_value = model.getObjective().getValue()
            subgradient = np.array([t.getValue() for t in relaxed_terms])

            self.print_verbose(j, model, dual_multipliers, subgradient)

            # Update best lower bound and mutlipliers
            if best_lower_bound < opt_value:
                best_lower_bound = opt_value
                best_multipliers = dual_multipliers
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            # Check Stopping criteria
            gradient_magnitude = np.linalg.norm(subgradient, 2)
            if gradient_magnitude <= self.tolerance:
                tolerance_reached = True
                self.print_iteration_info(j, opt_value, gradient_magnitude)
                break

            # Reduce step size parameter if lower bound does not improve for 10 iterations
            if no_improvement_counter == self.no_improvement_limit:
                step_size_parameter = step_size_parameter / 2
                no_improvement_counter = 0

            # Calculate new dual multipliers
            step_size = self.get_step_size("csl", gradient_magnitude)
            if not j == self.max_iterations - 1:
                dual_multipliers = dual_multipliers + step_size * subgradient

            self.print_iteration_info(j, opt_value, gradient_magnitude, step_size)

        stop_reason = "Tolerance" if tolerance_reached else "Max iterations"
        self.print_info(f"Subgradient Method finished ({stop_reason})")

        self.runtime_logger.log_task_end(
            f"subgradient_method_{self.n_calls}", subgradient_start_time
        )

        self.results.set_values(best_lower_bound, best_multipliers)
        return (model, self.results)

    def get_step_size(
        self,
        method="css",
        gradient_magnitude=None,
        step_size_parameter=None,
        upper_bound=None,
        opt_value=None,
    ):

        step_size = None

        if method == "css":
            # Constant step size
            step_size = self.const_step_size
        elif method == "csl" and gradient_magnitude != None:
            # Constant step length
            step_size = self.const_step_length / gradient_magnitude
        elif (
            method == "dss"
            and gradient_magnitude != None
            and step_size_parameter != None
            and upper_bound != None
            and opt_value != None
        ):
            # Dependent step size
            step_size = (
                step_size_parameter
                * (upper_bound - opt_value)
                / gradient_magnitude ** 2
            )
        else:
            raise ValueError("Incompatible arguments")

        return step_size

    def print_iteration_info(
        self,
        iteration: int,
        opt_value: float,
        gradient_magnitude: float,
        step_size: float = None,
    ):
        info = f"Iteration: {iteration} | Optimal value: {opt_value} | Gradient magnitude: {gradient_magnitude}"

        if step_size != None:
            info += f" | Step size: {step_size}"

        self.print_info(info)

    def print_info(self, text: str):
        if self.output_flag:
            print(text)

    def print_verbose(
        self, iteration: int, model: gp.Model, dual_multipliers: list, subgradient: list
    ):
        if self.output_verbose:
            print()
            print(f"Iteration {iteration}:")
            print("-" * 40)
            model.setParam("OutputFlag", 1)

            print()
            print(f"Dual multipliers: {dual_multipliers}")

            print()
            print("Model:")
            model.display()

            print()
            print("Optimal point:")
            model.printAttr("X")

            print()
            model.setParam("OutputFlag", 0)
            print(f"Subgradient: {subgradient}")
            print()

    def create_subgradient_log_dir(self, id: str):
        dir = os.path.join(self.log_dir, f"{id}_subgradient")
        os.mkdir(dir)
        return dir


class SolverResults:
    def __init__(self):
        self.obj_value = None
        self.multipliers = None

    def set_values(self, obj_value, multipliers):
        self.obj_value = obj_value
        self.multipliers = multipliers
