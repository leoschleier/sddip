import contextlib

import numpy as np


class Graph:
    def __init__(self, nodes: list, edges: list) -> None:
        self.edges = edges
        self.nodes = nodes

    def incidence_matrix(self):
        incidence_matrix = np.zeros((len(self.edges), len(self.nodes)))
        line_index = 0
        for edge in self.edges:
            incidence_matrix[line_index, edge[0] - 1] = 1
            incidence_matrix[line_index, edge[1] - 1] = -1
            line_index += 1
        return incidence_matrix


class Binarizer:
    def binary_expansion(self, x, upper_bound, precision):
        bin_multipliers = self.calc_binary_multipliers_from_precision(
            upper_bound, precision
        )

        return self.get_best_binary_approximation(x, bin_multipliers)

    def binary_expansion_from_n_binaries(self, x, upper_bound, n_binaries):
        bin_multipliers = self.calc_binary_multipliers_from_n_binaries(
            upper_bound, n_binaries
        )

        return self.get_best_binary_approximation(x, bin_multipliers)

    def binary_expansion_from_multipliers(self, x, binary_multipliers):
        return self.get_best_binary_approximation(x, binary_multipliers)

    def get_best_binary_approximation(self, x, binary_multipliers):
        lower_approximation_vars = self.calc_binary_lower_approximation(
            x, binary_multipliers
        )

        # None, if all binary variables of lower approximation equal 1
        upper_approximation_vars = self.calc_binary_upper_approximation(
            lower_approximation_vars
        )

        best_appr_vars = lower_approximation_vars

        # Choose approximation with the lowest error
        if upper_approximation_vars is not None:
            binary_multipliers = np.array(binary_multipliers)
            lower_approximation_vars = np.array(lower_approximation_vars)
            upper_approximation_vars = np.array(upper_approximation_vars)

            bin_appr_lower = lower_approximation_vars.dot(binary_multipliers)
            bin_appr_upper = upper_approximation_vars.dot(binary_multipliers)

            error_appr = x - bin_appr_lower
            error_appr_alt = bin_appr_upper - x

            if error_appr_alt < error_appr:
                best_appr_vars = upper_approximation_vars

        return (list(best_appr_vars), list(binary_multipliers))

    def calc_binary_multipliers_from_precision(
        self, upper_bound: float, precision: float
    ):
        n_bin_vars = int(np.log2(upper_bound / precision)) + 1

        return self.calc_binary_multipliers(precision, n_bin_vars)

    def calc_binary_multipliers_from_n_binaries(
        self, upper_bound: float, n_binaries: int
    ):
        precision = self.calc_precision_from_n_binaries(
            upper_bound, n_binaries
        )

        return self.calc_binary_multipliers(precision, n_binaries)

    def calc_binary_multipliers(self, precision: float, n_binaries: int):
        return [precision * 2**i for i in range(n_binaries)]

    def calc_precision_from_n_binaries(
        self, upper_bound: float, n_binaries: int
    ):
        return upper_bound / (sum([2 ** (k) for k in range(n_binaries)]))

    def calc_max_abs_error(self, precision: int):
        if 0 > precision > 1:
            msg = "Precision should be between 0 and 1."
            raise ValueError(msg)
        return precision / 2

    def calc_binary_lower_approximation(
        self, x: float, binary_multipliers: list
    ):
        bin_vars = []

        for b in reversed(binary_multipliers):
            v = 0
            if x >= b:
                v = 1
                x -= b
            bin_vars.insert(0, v)

        return bin_vars

    def calc_binary_upper_approximation(self, lower_approximation_vars):
        first_zero_index = None

        with contextlib.suppress(ValueError):
            first_zero_index = lower_approximation_vars.index(0)

        upper_approximation_vars = None

        if first_zero_index is not None:
            upper_approximation_vars = lower_approximation_vars.copy()
            upper_approximation_vars[:first_zero_index] = [
                0
            ] * first_zero_index
            upper_approximation_vars[first_zero_index] = 1

        return upper_approximation_vars
