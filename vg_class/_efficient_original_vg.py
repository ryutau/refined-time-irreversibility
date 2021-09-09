import networkx as nx
from ts2vg import NaturalVisibilityGraph

from ._base_graphs import BaseGraph


class EfficientVisibilityGraph(BaseGraph):
    def __init__(self, time_series, name, window_width):
        self.window_width = window_width
        super().__init__(time_series, name)

    def make_graph(self):
        vg = NaturalVisibilityGraph([i[1] for i in self.ts])
        G = nx.DiGraph()
        node_dic = [(node[0], {"value": node[1]}) for node in self.ts]
        G.add_nodes_from(node_dic)
        if self.window_width is None:
            edge_list = [sorted([d + 1 for d in edge]) for edge in vg.edgelist()]
        else:
            edge_list = [
                sorted((edge[0] + 1, edge[1] + 1))
                for edge in vg.edgelist()
                if abs(edge[0] - edge[1]) <= self.window_width
            ]
        del vg
        G.add_edges_from(edge_list)
        return G
