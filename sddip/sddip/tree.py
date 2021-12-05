class ProblemParameters:

    def __init__(self, params=None):
        self.params = {} if params == None else params


class RecombiningTree(ProblemParameters):

    def __init__(self, n_nodes_per_stage: list, params=None):
        super(RecombiningTree, self).__init__(params)
        self.n_stages = len(n_nodes_per_stage)
        self.stages = [Stage(i, n_nodes_per_stage[i])
                       for i in range(self.n_stages)]
        self.stage_index = 0

    def __iter__(self):
        return IterableWrapper(self.stages)

    def __reversed__(self):
        return IterableWrapper(self.stages, True)

    def get_stage(self, stage_num: int):
        return self.stages[stage_num]

    def get_node(self, stage_num: int, node_num: int):
        return self.get_stage(stage_num).get_node(node_num)


class Stage(ProblemParameters):

    def __init__(self, stage_num, n_nodes, params=None):
        super(Stage, self).__init__(params)
        self.stage_num = stage_num
        self.nodes = [Node(i, self.stage_num) for i in range(n_nodes)]
        self.cut_gradients = []
        self.cut_intercepts = []

    def __iter__(self):
        return IterableWrapper(self.nodes)

    def __reversed__(self):
        return IterableWrapper(self.nodes, True)

    def get_node(self, node_num: int):
        return self.nodes[node_num]


class Node(ProblemParameters):

    def __init__(self, node_num, stage_num, prob=0., params=None):
        super(Node, self).__init__(params)
        self.node_num = node_num
        self.stage_num = stage_num
        self.prob = prob


class IterableWrapper:

    def __init__(self, iterable, reversed=False):
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
