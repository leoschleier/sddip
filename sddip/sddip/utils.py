import numpy as np

class Graph:

    def __init__(self, nodes:list, edges:list):
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

