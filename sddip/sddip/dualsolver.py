import copy
import logging
import os
from abc import ABC, abstractmethod
from time import time
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np

from . import sddip_logging

logger = logging.getLogger(__name__)


class DualSolver(ABC):
    def __init__(
        self, max_iterations: int, tolerance: float, log_dir: str, tag: str
    ):
        self.tag = tag

        log_manager = sddip_logging.LogManager()
        runtime_log_dir = log_manager.create_log_dir(f"{self.tag}_log")
        self.runtime_logger = sddip_logging.RuntimeLogger(runtime_log_dir)

        self.output_flag = False
        self.output_verbose = False

        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.log_dir = log_dir
        self.n_calls = 0
        self.solver_time = 0
        self.start_time = None

        self.results = SolverResults()

    @abstractmethod
    def solve(self):
        """Solve the dual problem."""

    def get_subgradient_and_value(
        self, model, objective_terms, relaxed_terms, dual_multipliers
    ) -> Tuple[np.ndarray, float]:
        """Compute the subgradient and the objective value of the dual
        problem.
        """
        gradient_len = len(relaxed_terms)

        total_objective = objective_terms + gp.quicksum(
            relaxed_terms[i] * dual_multipliers[i] for i in range(gradient_len)
        )

        model.setObjective(total_objective)

        model.update()

        model.optimize()

        self.solver_time += model.Runtime

        subgradient = np.array([t.getValue() for t in relaxed_terms])
        opt_value = model.getObjective().getValue()

        return (subgradient, opt_value)

    def on_solver_call(self):
        self.n_calls += 1
        self.solver_time = 0
        self.log_task_start()

    def log_method_finished(
        self,
        stop_reason: str,
        iteration: int,
        lowest_gradient_magnitude: float,
        best_lower_bound: float,
        method: str = "",
        n_serious_steps: Optional[int] = None,
    ):
        if n_serious_steps is not None:
            n_null_steps = iteration - n_serious_steps
            steps = f", ns/ss: {n_null_steps}/{n_serious_steps}"
        else:
            steps = ""

        logger.info(
            "Dual solver finished (%s, m: %s, i: %s%s, st: %.3f, "
            "g: %.3f, lb: %.3f)",
            stop_reason,
            self.tag + method,
            iteration,
            steps,
            self.solver_time,
            lowest_gradient_magnitude,
            best_lower_bound,
        )

    def log_task_start(self):
        self.start_time = time()

    def log_task_end(self):
        self.runtime_logger.log_task_end(
            f"{self.tag}_{self.n_calls}", self.start_time
        )


class SolverResults:
    def __init__(self):
        self.obj_value = None
        self.multipliers = None
        self.solver_time = None
        self.n_iterations = None

    def set_values(self, obj_value, multipliers, n_iterations, solver_time):
        self.obj_value = obj_value
        self.multipliers = multipliers
        self.n_iterations = n_iterations
        self.solver_time = solver_time


class SubgradientMethod(DualSolver):

    TAG = "SG"

    def __init__(
        self,
        max_iterations: int,
        tolerance: float,
        log_dir: str = None,
    ) -> None:
        super().__init__(max_iterations, tolerance, log_dir, self.TAG)

        # Step size parameters
        self.initial_lower_bound = 0
        self.const_step_size = 1
        self.const_step_length = 1
        self.step_size_parameter = 2
        self.no_improvement_limit = 10
        self.smoothing_factor = 0.25
        self.log_flag = False

    def solve(
        self,
        model: gp.Model,
        objective_terms,
        relaxed_terms,
        optimal_value_estimate: float = None,
        log_id: str = None,
    ) -> Tuple[gp.Model, SolverResults]:

        self.on_solver_call()
        model.setParam("OutputFlag", 0)

        if self.log_flag:
            current_log_dir = self.create_subgradient_log_dir(log_id)
            gurobi_logger = sddip_logging.GurobiLogger(current_log_dir)

        self.results = SolverResults()

        gradient_len = len(relaxed_terms)
        dual_multipliers = np.zeros(gradient_len)
        # dual_multipliers = np.full(gradient_len, 10000)

        best_lower_bound = self.initial_lower_bound
        best_multipliers = dual_multipliers
        lowest_gm = 100

        no_improvement_counter = 0

        step_size_parameter = self.step_size_parameter
        step_size = None

        tolerance_reached = False

        logger.debug("Subgradient Method started")

        for j in range(self.max_iterations):

            # Optimize
            subgradient, opt_value = self.get_subgradient_and_value(
                model, objective_terms, relaxed_terms, dual_multipliers
            )

            if self.log_flag:
                gurobi_logger.log_model(
                    model, str(j).zfill(len(str(self.max_iterations)))
                )

            self.log_verbose(j, model, dual_multipliers, subgradient)

            gradient_magnitude = np.linalg.norm(subgradient, 2)

            lowest_gm = (
                gradient_magnitude
                if gradient_magnitude < lowest_gm
                else lowest_gm
            )

            # Update best lower bound and mutlipliers
            if best_lower_bound < opt_value:
                best_lower_bound = opt_value
                best_multipliers = dual_multipliers
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            # Check Stopping criteria
            if gradient_magnitude <= self.tolerance:
                tolerance_reached = True
                self.log_iteration_info(j, opt_value, gradient_magnitude)
                break
            if optimal_value_estimate:
                if abs(optimal_value_estimate - opt_value) <= 10 ** (-8):
                    tolerance_reached = True
                    break

            # Reduce step size parameter if lower bound does not improve
            # for 10 iterations
            if no_improvement_counter == self.no_improvement_limit:
                step_size_parameter = step_size_parameter / 2
                no_improvement_counter = 0

            # Calculate new dual multipliers
            method = "dss" if optimal_value_estimate else "csl"
            # method = "csl"
            step_size = self.get_step_size(
                method, subgradient, opt_value, optimal_value_estimate
            )

            if not j == self.max_iterations - 1:
                dual_multipliers = dual_multipliers + step_size * subgradient

            self.log_iteration_info(j, opt_value, subgradient, step_size)

        stop_reason = "Tolerance" if tolerance_reached else "Max iterations"

        self.log_method_finished(
            stop_reason, j + 1, lowest_gm, best_lower_bound, method
        )

        self.log_task_end()

        self.results.set_values(
            best_lower_bound, best_multipliers, j + 1, self.solver_time
        )
        return (model, self.results)

    def get_step_size(
        self,
        method: str = "css",
        gradient: list = [None],
        function_value: float = None,
        opt_value_estimate: float = None,
    ):

        step_size = None
        gradient_magnitude = (
            np.linalg.norm(gradient, 2) if any(gradient) else None
        )

        if method == "css":
            # Constant step size
            step_size = self.const_step_size
        elif method == "csl" and gradient_magnitude != None:
            # Constant step length
            step_size = self.const_step_length / gradient_magnitude
        elif (
            method == "dss"
            and gradient_magnitude != None
            and function_value != None
            and opt_value_estimate != None
        ):
            # Dependent step size
            # smoothed_gradient = (
            #     1 - self.smoothing_factor
            # ) * gradient + self.smoothing_factor * smoothed_gradient

            step_size = (
                opt_value_estimate - function_value
            ) / gradient_magnitude**2

        else:
            raise ValueError("Incompatible arguments")

        return step_size

    def log_iteration_info(
        self,
        iteration: int,
        opt_value: float,
        gradient_magnitude: float,
        step_size: float = None,
    ):
        logger.info(
            "Iteration: %s | Optimal value: %s | Gradient magnitude: %s | "
            "Step size: %s",
            iteration,
            opt_value,
            gradient_magnitude,
            step_size,
        )

    def log_verbose(
        self,
        iteration: int,
        model: gp.Model,
        dual_multipliers: list,
        subgradient: list,
    ):

        logger.debug("Iteration %s:", iteration)

        logger.debug("Dual multipliers: %s", dual_multipliers)
        logger.debug("Subgradient: %s", subgradient)

        # TODO Log model and optimal point
        # model.setParam("OutputFlag", 1)
        # Model model.display()
        # Optimal point: model.printAttr("X")
        # model.setParam("OutputFlag", 0)

    def create_subgradient_log_dir(self, id: str):
        sg_dir = os.path.join(self.log_dir, f"{id}_subgradient")
        os.mkdir(sg_dir)
        return dir


class BundleMethod(DualSolver):

    TAG = "BM"
    ABS_PREDICTED_ASCENT = "abs"
    REL_PREDICTED_ASCENT = "rel"

    def __init__(
        self,
        max_iterations: int,
        tolerance: float,
        log_dir: str,
        predicted_ascent="abs",
    ):
        super().__init__(max_iterations, tolerance, log_dir, self.TAG)

        self.u_init = 1
        self.u_min = 0.1
        self.m_l = 0.3
        self.m_r = 0.7

        self.predicted_ascent = predicted_ascent

        if predicted_ascent == self.ABS_PREDICTED_ASCENT:
            self._get_predicted_ascent = self._absolute_predicted_ascent
        elif predicted_ascent == self.REL_PREDICTED_ASCENT:
            self._get_predicted_ascent = self._relative_predicted_ascent
        else:
            raise ValueError(
                "Argument predicted_ascent is "
                f"'{self.predicted_ascent}'. It must either be "
                f"'{self.ABS_PREDICTED_ASCENT}' or "
                f"'{self.REL_PREDICTED_ASCENT}'."
            )

    def _absolute_predicted_ascent(
        self, current_value: float, new_value: float
    ) -> float:
        """Compute absolute predicted ascent."""
        return max(new_value - current_value, 0)

    def _relative_predicted_ascent(
        self, current_value: float, new_value: float
    ) -> float:
        """Compute relative predicted ascent."""
        return max((new_value - current_value) / current_value, 0)

    def solve(
        self, model: gp.Model, objective_terms, relaxed_terms
    ) -> Tuple[gp.Model, SolverResults]:
        """Solve the dual problem using the bundle method."""

        logger.debug("Bundle method started")

        self.on_solver_call()
        model.setParam("OutputFlag", 0)

        tolerance_reached = False
        u = self.u_init
        i_u = 0
        var_est = 10**8

        n_serious_steps = 0

        gradient_len = len(relaxed_terms)
        x_new = np.zeros(gradient_len)
        x_best = np.zeros(gradient_len)

        # Initial subgradient and best lower bound
        subgradient, f_best = self.get_subgradient_and_value(
            model, objective_terms, relaxed_terms, x_best
        )

        f_new = f_best

        # Lowest known gradient magnitude
        lowest_gm = np.linalg.norm(np.array(subgradient))

        # Subproblem with cutting planes
        subproblem, v, x = self.create_subproblem(gradient_len)

        for i in range(self.max_iterations):

            # Add new plane to subproblem
            new_plane = f_new + gp.quicksum(
                subgradient[j] * (x[j] - x_new[j]) for j in range(gradient_len)
            )

            obj = v - u / 2 * gp.quicksum(
                (x[j] - x_best[j]) ** 2 for j in range(gradient_len)
            )
            subproblem.setObjective(obj, gp.GRB.MAXIMIZE)
            subproblem.addConstr(v <= new_plane, name=f"{i+1}")
            subproblem.update()

            # Solve subproblem
            subproblem.optimize()

            # Candidate dual multipliers
            x_new = [x[j].x for j in range(gradient_len)]

            # Candidate optimal value
            subgradient, f_new = self.get_subgradient_and_value(
                model, objective_terms, relaxed_terms, x_new
            )

            # Predicted ascent
            delta = self._get_predicted_ascent(f_best, v.x)

            # Update lowest known gradient magnitude for logging purposes
            lowest_gm = min(lowest_gm, np.linalg.norm(np.array(subgradient)))

            serious_step = f_new - f_best >= self.m_l * delta
            # Weight update
            # u, i_u, var_est = self.weight_update(
            #     u,
            #     i_u,
            #     var_est,
            #     x_new,
            #     f_new,
            #     x_best,
            #     f_best,
            #     v.x,
            #     subgradient,
            #     serious_step,
            # )
            if serious_step:
                # Serious step
                logger.debug(
                    "Serious step: i = %s, f_new = %s, "
                    "f_best = %s, lowest_gm = %s",
                    i + 1,
                    f_new,
                    f_best,
                    lowest_gm,
                )
                x_best = copy.copy(x_new)
                f_best = copy.copy(f_new)
                n_serious_steps += 1

            # Check stopping criterion
            if delta <= self.tolerance:
                tolerance_reached = True
                break

        stop_reason = "Tolerance" if tolerance_reached else "Max iterations"

        self.log_task_end()

        self.log_method_finished(
            stop_reason,
            i + 1,
            lowest_gm,
            f_best,
            n_serious_steps=n_serious_steps,
        )

        self.results.set_values(
            f_best, np.array(x_best), i + 1, self.solver_time
        )

        return (model, self.results)

    def weight_update(
        self,
        u_current,
        i_u,
        var_est,
        x_new,
        f_new,
        x_best,
        f_best,
        f_hat,
        subgradient,
        serious_step,
    ):
        """Update the weight for the bundle method with proximity
        control.
        """
        variation_estimate = var_est

        delta = f_hat - f_best
        u_int = 2 * u_current * (1 - (f_new - f_best) / delta)
        u = u_current
        # print(f"f: {f_new - f_best}")
        if serious_step:
            weight_too_large = (f_new - f_best) >= (self.m_r * delta)
            if weight_too_large and i_u > 0:
                u = u_int
            elif i_u > 3:
                u = u_current / 2
            u_new = max(u, u_current / 10, self.u_min)
            # print(f"u: {u}, {u_current/10}, {self.u_min}")
            variation_estimate = max(variation_estimate, 2 * delta)
            i_u = max(i_u + 1, 1) if u_new == u_current else 1
            # Exit
        else:
            # print("Null")
            p = -u_current * (np.array(x_new) - np.array(x_best))
            alpha = delta - np.linalg.norm(p, ord=2) ** 2 / u_current
            variation_estimate = min(
                variation_estimate, np.linalg.norm(p, ord=1) + alpha
            )
            # x_best x_new Reihenfolge ?
            linearization_error = (
                f_new
                + np.array(subgradient).dot(np.array(x_best) - np.array(x_new))
                - f_best
            )
            if (
                linearization_error > max(variation_estimate, 10 * delta)
                and i_u < -3
            ):
                u = u_int
            u_new = min(u, 10 * u_current)
            i_u = min(i_u - 1, -1) if u_new == u_current else -1
            # Exit

        return u_new, i_u, variation_estimate

    def create_subproblem(
        self, n_dual_multipliers: int
    ) -> Tuple[gp.Model, gp.Var, gp.tupledict]:
        """Create the bundle method's subproblem."""

        subproblem = gp.Model("Subproblem")
        subproblem.setParam("OutputFlag", 0)
        v = subproblem.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="v"
        )
        x = subproblem.addVars(
            n_dual_multipliers,
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            name="x",
        )
        return subproblem, v, x
