class Node:
    def __init__(
        self, stage: int, index: int, realization: int, parent: "Node" = None
    ) -> None:
        self.stage = stage
        self.index = index
        self.realization = realization
        self.parent = parent
        self.children = []

    def set_children(self, children: list) -> None:
        self.children = children

    def get_ancestors(self, horizon: int | None = None):
        n_ancestors = horizon if horizon else self.stage
        ancestors = []

        current_node = self
        for _ in range(n_ancestors):
            current_node = current_node.parent
            ancestors.append(current_node)

        return ancestors


class ScenarioTree:
    def __init__(self, n_realizations_per_stage: list[int]) -> None:
        self.n_stages = len(n_realizations_per_stage)
        self.n_nodes_per_stage = [1]
        self.root = Node(0, 0, 0)
        self.nodes = [[self.root]]
        self._build_tree(n_realizations_per_stage)

    def _build_tree(self, n_realizations_per_stage: list[int]) -> None:
        for t in range(1, self.n_stages):
            node_index = 0
            stage_nodes = []
            for n in self.nodes[t - 1]:
                children = []
                for r in range(n_realizations_per_stage[t]):
                    children.append(Node(t, node_index, r, n))
                    node_index += 1
                n.set_children(children)
                stage_nodes += children
            self.nodes.append(stage_nodes)

        self.n_nodes_per_stage = [len(s) for s in self.nodes]

    def __str__(self) -> str:
        total_number_of_nodes = sum(self.n_nodes_per_stage)
        return f"ScenarioTree: Stages = {self.n_stages}, Nodes = {total_number_of_nodes}"

    def get_node(self, stage, index):
        return self.nodes[stage][index]

    def get_stage_nodes(self, stage):
        return self.nodes[stage]


class Stage:
    def __init__(self, stage_num, n_nodes, params=None) -> None:
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


class IterableWrapper:
    def __init__(self, iterable, reversed=False) -> None:
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
