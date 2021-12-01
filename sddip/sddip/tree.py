
class RecombiningTree:
    
    def __init__(self, n_nodes_per_stage:list):
        self.n_stages = len(n_nodes_per_stage)
        self.stages = [Stage(i, n_nodes_per_stage[i]) for i in range(self.n_stages)]
        self.stage_index = 0

    def get_stage(self, stage_num:int):
        return self.stages[stage_num]
    
    def current(self):
        return self.stages(self.stage_index)
    
    def next(self):
        if self.stage_index == len(self.stages-1):
            raise IndexError("Stage index out of bounds.")
        self.stage_index += 1
        return self.current()
    
    def prev(self):
        if self.stage_index == 0:
            raise IndexError("Stage index out of bounds.")
        self.stage_index -= 1
        return self.current()


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

    
