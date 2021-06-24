from ._base_graphs import BaseGraph
from ._refined_vg import RefinedVG


class BasicSubGraph(BaseGraph):
    def __init__(self, full_graph, N):
        self.name = full_graph.name
        self.N = N
        self.ts = full_graph.ts[:N]
        self.window_width = full_graph.window_width
        self.graph = full_graph.graph.subgraph(range(1, N + 1))

    def make_graph(self):
        pass


class RefinedSubGraph(RefinedVG):
    def __init__(self, full_graph, N):
        self.name = full_graph.name
        self.window_width = full_graph.window_width
        self.N = N
        self.ts = full_graph.ts[:N]
        self.graph = full_graph.graph.subgraph(range(1, N + 1))

    def make_graph(self):
        pass
