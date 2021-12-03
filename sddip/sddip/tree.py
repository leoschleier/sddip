
class RecombiningTree:
    
    def __init__(self, n_nodes_per_stage:list):
        self.n_stages = len(n_nodes_per_stage)
        self.stages = [Stage(i, n_nodes_per_stage[i]) for i in range(self.n_stages)]
        self.stage_index = 0

    def __iter__(self):
        return IterableWrapper(self, self.stages)
    
    def __reversed__(self):
        return IterableWrapper(self, self.stages, True)

    def get_stage(self, stage_num:int):
        return self.stages[stage_num]


class Stage:
    
    def __init__(self, stage_num, n_nodes):
        self.stage_num = stage_num
        self.n_nodes = n_nodes
        self.nodes = [Node(i, self.stage_num) for i in range(n_nodes)]
        self.cut_gradients = []
        self.cut_intercepts = []
        self.det_params = {}


class Node:

    def __init__(self, node_num, stage_num, prob=0.):
        self.node_num = node_num
        self.stage_num = stage_num
        self.prob = prob
        self.stoch_params = {}

    
class IterableWrapper:

    def __init__(slef, iterable, reversed = False):
        self.iterable = iterable
        self.reversed = reversed
        self.index = -1 if reversed else 0

    def __next__(self):
        try:
            result = self.iterable[self.index]
        except IndexError:
            raise StopIteration
        self.index += -1 if self.reversed else +1
        return result
