import numpy as np


class Graph:

    def __init__(self, nodes: list, edges: list):
        self.edges = edges
        self.nodes = nodes

    def incidence_matrix(self):
        incidence_matrix = np.zeros((len(self.edges), len(self.nodes)))
        line_index = 0
        for edge in self.edges:
            incidence_matrix[line_index, edge[0]-1] = 1
            incidence_matrix[line_index, edge[1]-1] = -1
            line_index += 1
        return incidence_matrix


class Binarizer:

    def binary_expansion(self, x, upper_bound, precision):
        bin_multipliers = self.calc_binary_multipliers(upper_bound, precision)

        lower_approximation_vars = self.calc_binary_lower_approximation(
            x, bin_multipliers)

        # None, if all binary variables of lower approximation equal 1
        upper_approximation_vars = self.calc_binary_upper_approximation(
            lower_approximation_vars)

        best_appr_vars = lower_approximation_vars

        # Choose approximation with the lowest error
        if upper_approximation_vars != None:
            bin_multipliers = np.array(bin_multipliers)
            lower_approximation_vars = np.array(lower_approximation_vars)
            upper_approximation_vars = np.array(upper_approximation_vars)

            bin_appr_lower = lower_approximation_vars.dot(bin_multipliers)
            bin_appr_upper = upper_approximation_vars.dot(bin_multipliers)

            error_appr = x - bin_appr_lower
            error_appr_alt = bin_appr_upper - x

            if error_appr_alt < error_appr:
                best_appr_vars = upper_approximation_vars

        return (list(best_appr_vars), list(bin_multipliers))

    def calc_binary_multipliers(self, upper_bound, precision):
        n_bin_vars = int(np.log2(upper_bound/precision))+1
        bin_multipliers = [precision*2**i for i in range(n_bin_vars)]

        return bin_multipliers

    def calc_binary_lower_approximation(self, x, binary_multipliers):
        bin_vars = []

        for b in reversed(binary_multipliers):
            v = 0
            if x >= b:
                v = 1
                x -= 1
            bin_vars.insert(0, v)

        return bin_vars

    def calc_binary_upper_approximation(self, lower_approximation_vars):
        first_zero_index = None

        try:
            first_zero_index = lower_approximation_vars.index(0)
        except ValueError:
            pass

        upper_approximation_vars = None

        if first_zero_index != None:
            upper_approximation_vars = lower_approximation_vars.copy()
            upper_approximation_vars[:first_zero_index] = [0]*first_zero_index
            upper_approximation_vars[first_zero_index] = 1

        return upper_approximation_vars
