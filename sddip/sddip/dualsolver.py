import gurobipy as gp
import numpy as np


class SubgradientMethod:
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 10 ** (-3),
        initial_lower_bound: float = 0,
    ) -> None:
        self.inital_lower_bound = initial_lower_bound
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.results = SolverResults()
        self.output_flag = False

        # Step size parameters
        self.const_step_size = 1
        self.const_step_length = 1
        self.step_size_parameter = 2
        self.no_improvement_limit = 10

    def solve(
        self, model: gp.Model, objective_terms, relaxed_terms, upper_bound: float = None
    ) -> gp.Model:

        model.setParam("OutputFlag", 0)

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

            # Optimal value
            opt_value = model.getObjective().getValue()
            subgradient = np.array([t.getValue() for t in relaxed_terms])

            # Update best lower bound and mutlipliers
            if best_lower_bound <= opt_value:
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


class SolverResults:
    def __init__(self):
        self.obj_value = None
        self.multipliers = None

    def set_values(self, obj_value, multipliers):
        self.obj_value = obj_value
        self.multipliers = multipliers
