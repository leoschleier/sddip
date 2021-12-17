import gurobipy as gp
import numpy as np


class SubgradientMethod:

    def __init__(self, initial_lower_bound = 0, max_iterations = 1000, tolerance = 10**(-3)) -> None:
        self.inital_lower_bound = initial_lower_bound
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.results = SolverResults()

    def solve(self, model: gp.Model, objective_terms, relaxed_terms, upper_bound: float) -> gp.Model:
        
        self.results = SolverResults()

        gradient_len = len(relaxed_terms)
        dual_multipliers = np.zeros(gradient_len)

        lower_bound = self.inital_lower_bound
        no_improvement_counter = 0
        step_size_parameter = 2
        step_size = None

        for j in range(self.max_iterations):
            # Update model
            total_objective = objective_terms \
                + gp.quicksum(relaxed_terms[i]*dual_multipliers[i] for i in range(gradient_len))
            model.setObjective(total_objective)
            model.update()

            # Optimization
            model.optimize()

            # Optimal value
            opt_value = model.getObjective().getValue()
            subgradient = np.array([t.getValue() for t in relaxed_terms])

            # Check Stopping criteria
            gradient_magnitude = np.linalg.norm(subgradient, 2)
            if gradient_magnitude <= self.tolerance: break

            # Update lower bound
            new_lower_bound = max(lower_bound, opt_value)
            if lower_bound == new_lower_bound: no_improvement_counter +=1
            self.results.lower_bound = lower_bound
            
            # Reduce step size if lower bound does not improve for 10 iterations
            if no_improvement_counter == 10:
                step_size_parameter = step_size_parameter/2
                no_improvement_counter = 0

            # Calculate new dual multipliers
            if not j == self.max_iterations-1:
                step_size = step_size_parameter*(upper_bound-opt_value)/gradient_magnitude**2
                dual_multipliers = dual_multipliers + step_size*subgradient

        self.results.set_values(opt_value, dual_multipliers, subgradient, lower_bound, step_size)

        return (model, self.results)


class SolverResults:

    def __init__(self):
        self.obj_value = None
        self.multipliers = None
        self.subgradient = None
        self.lower_bound = None
        self.step_size = None

    def set_values(self, obj_value, multipliers, subgradient, lower_bound, step_size):
        self.obj_value =obj_value
        self.multipliers = multipliers
        self.subgradient = subgradient
        self.lower_bound = lower_bound
        self.step_size = step_size
