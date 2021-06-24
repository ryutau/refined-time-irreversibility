from itertools import combinations

import networkx as nx

from ._base_graphs import BaseGraph


class RiseVisibilityGraph(BaseGraph):
    def __init__(self, time_series, name, window_width):
        self.window_width = window_width
        super().__init__(time_series, name)

    def make_graph(self):
        # Initialize directed graph.
        G = nx.DiGraph()
        # Add all nodes to the graph.
        node_dic = [(node[0], {"value": node[1]}) for node in self.ts]
        G.add_nodes_from(node_dic)
        # Check every pair of the nodes and add an array if it's connected.
        for (ta, ya), (tb, yb) in combinations(self.ts, 2):
            # Every pair fulfills ta < tb from its construction.
            if ((tb - ta) <= self.window_width) & (ya < yb):
                slope = (yb - ya) / (tb - ta)
                connect = True
                # Check every node c which is located between a and b.
                for (tc, yc) in self.ts[ta : tb - 1]:
                    if yc >= ya + slope * (tc - ta):
                        connect = False
                        break
                if connect:
                    G.add_edge(ta, tb)
        return G


class FallVisibilityGraph(BaseGraph):
    def __init__(self, time_series, name, window_width):
        self.window_width = window_width
        super().__init__(time_series, name)

    def make_graph(self):
        G = nx.DiGraph()
        node_dic = [(node[0], {"value": node[1]}) for node in self.ts]
        G.add_nodes_from(node_dic)
        for (ta, ya), (tb, yb) in combinations(self.ts, 2):
            if ((tb - ta) <= self.window_width) & (ya > yb):
                slope = (yb - ya) / (tb - ta)
                connect = True
                for (tc, yc) in self.ts[ta : tb - 1]:
                    if yc >= ya + slope * (tc - ta):
                        connect = False
                        break
                if connect:
                    G.add_edge(ta, tb)
        return G


class RiseInvisibilityGraph(BaseGraph):
    def __init__(self, time_series, name, window_width):
        self.window_width = window_width
        super().__init__(time_series, name)

    def make_graph(self):
        G = nx.DiGraph()
        node_dic = [(node[0], {"value": node[1]}) for node in self.ts]
        G.add_nodes_from(node_dic)
        for (ta, ya), (tb, yb) in combinations(self.ts, 2):
            if ((tb - ta) <= self.window_width) & (ya < yb):
                slope = (yb - ya) / (tb - ta)
                connect = True
                for (tc, yc) in self.ts[ta : tb - 1]:
                    if yc <= ya + slope * (tc - ta):
                        connect = False
                        break
                if connect:
                    G.add_edge(ta, tb)
        return G


class FallInvisibilityGraph(BaseGraph):
    def __init__(self, time_series, name, window_width):
        self.window_width = window_width
        super().__init__(time_series, name)

    def make_graph(self):
        G = nx.DiGraph()
        node_dic = [(node[0], {"value": node[1]}) for node in self.ts]
        G.add_nodes_from(node_dic)
        for (ta, ya), (tb, yb) in combinations(self.ts, 2):
            if ((tb - ta) <= self.window_width) & (ya > yb):
                slope = (yb - ya) / (tb - ta)
                connect = True
                for (tc, yc) in self.ts[ta : tb - 1]:
                    if yc <= ya + slope * (tc - ta):
                        connect = False
                        break
                if connect:
                    G.add_edge(ta, tb)
        return G
