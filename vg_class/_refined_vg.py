from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from tools.convenient_functions import KLD
from tools.save import save_dir

from ._base_graphs import BaseGraph

plt.rcParams["font.size"] = 15
plt.rcParams["axes.formatter.use_mathtext"] = True


class RefinedVG(BaseGraph):
    def __init__(self, time_series, name="temporal DVG", window_width=10):
        self.window_width = window_width
        super().__init__(time_series, name)

    def make_graph(self):
        # Initialize directed graph.
        G = nx.DiGraph()
        # Add all nodes to the graph.
        node_dic = [(node[0], {"value": node[1]}) for node in self.ts]
        G.add_nodes_from(node_dic)
        # Check neighbors from every node.
        # The size of iteration here is len(time_series).
        for (ta, ya) in self.ts:
            # The size of iteration here is window_width.
            for i in range(min(self.window_width, len(self.ts) - ta)):
                tb = ta + i + 1
                yb = self.ts[tb - 1][1]
                slope = (yb - ya) / (i + 1)
                real_sequence = [self.ts[ta + j][1] for j in range(i)]
                line_sequence = [ya + (slope * j) for j in range(1, i + 1)]
                # Check each conditions.
                isRise = slope > 0
                isFall = slope <= 0
                isVisible = all(
                    real < line
                    for real, line in zip(real_sequence, line_sequence)
                )
                isInvisible = all(
                    # real > line
                    real >= line
                    for real, line in zip(real_sequence, line_sequence)
                )
                if isVisible | isInvisible:
                    G.add_edge(
                        ta,
                        tb,
                        edge_kind=self.ret_edge_kind(
                            isRise, isFall, isVisible, isInvisible
                        ),
                    )
        return G

    def ret_edge_kind(self, isRise, isFall, isVisible, isInvisible):
        if isRise & isVisible:
            return "RV"
        elif isRise & isInvisible:
            return "RIV"
        elif isFall & isVisible:
            return "FV"
        elif isFall & isInvisible:
            return "FIV"
        else:
            raise ValueError

    def get_degree_sequence(self, *, edge_kind, deg_kind):
        kind_edge_list = [
            (f, t)
            for f, t, prop in self.graph.edges.data()
            if prop["edge_kind"] == edge_kind
        ]
        if len(kind_edge_list) == 0:
            degree_sequence = {i: 0 for i in range(1, self.N + 1)}
        else:
            outdegrees, indegrees = zip(*kind_edge_list)
            if deg_kind == "degree":
                degree_sequence = Counter(indegrees) + Counter(outdegrees)
            elif deg_kind == "indegree":
                degree_sequence = Counter(indegrees)
            elif deg_kind == "outdegree":
                degree_sequence = Counter(outdegrees)
            else:
                raise ValueError(
                    "deg_kind should either be degree, indegree or outdegree"
                )
            degree_sequence = {
                i: degree_sequence[i] for i in range(1, self.N + 1)
            }
        return degree_sequence

    def get_pattern_sequence_table(self, deg_kind):
        pattern_sequence_table = pd.concat(
            [
                pd.Series(
                    self.get_degree_sequence(
                        edge_kind=edge_kind, deg_kind=deg_kind
                    ),
                    name=edge_kind,
                )
                for edge_kind in ["RV", "RIV", "FV", "FIV"]
            ],
            axis=1,
        )
        pattern_sequence_table.loc[
            :, "pattern_code"
        ] = pattern_sequence_table.apply(
            self.to_pattern, axis=1, deg_kind=deg_kind
        )
        return pattern_sequence_table

    def to_pattern(self, data, deg_kind):
        if deg_kind == "indegree":
            pattern_code = int(
                data.RV * 1e3 + data.RIV * 1e2 + data.FV * 1e1 + data.FIV
            )
        elif deg_kind == "outdegree":
            pattern_code = int(
                data.FV * 1e3 + data.FIV * 1e2 + data.RV * 1e1 + data.RIV
            )
        return pattern_code

    def get_pattern_dict(self, deg_kind, start=None, end=None):
        pattern_seq = self.get_pattern_sequence_table(
            deg_kind=deg_kind
        ).pattern_code
        if (start is None) and (end is None):
            if deg_kind == "indegree":
                pattern_cnt = pattern_seq.iloc[
                    self.window_width :
                ].value_counts()
            elif deg_kind == "outdegree":
                pattern_cnt = pattern_seq.iloc[
                    : -self.window_width
                ].value_counts()
        else:
            if deg_kind == "indegree":
                pattern_cnt = pattern_seq.iloc[
                    start - 1 + self.window_width : end - 1
                ].value_counts()
            elif deg_kind == "outdegree":
                pattern_cnt = pattern_seq.iloc[
                    start - 1 : end - 1 - self.window_width
                ].value_counts()
        pattern_dict = defaultdict(lambda: 0, pattern_cnt.to_dict())
        return pattern_dict

    def compute_irreversibility(self, start=None, end=None):
        inpattern_dict = self.get_pattern_dict("indegree", start, end)
        outpattern_dict = self.get_pattern_dict("outdegree", start, end)
        kld = KLD(inpattern_dict, outpattern_dict)
        return kld

    def visualize_in_out_diff(self, dir_path=None, file_name=None):
        inpattern_dict = self.get_pattern_dict("indegree")
        outpattern_dict = self.get_pattern_dict("outdegree")
        diff_dict = {
            k: v - outpattern_dict[k] for k, v in inpattern_dict.items()
        }
        irr = round(self.compute_irreversibility(), 5)
        plt.figure(figsize=(12, 6))
        plt.suptitle("Detailed topological pattern difference", fontsize=20)
        plt.title(
            rf"{self.name}, $\omega={self.window_width}$: $\mathcal{{D}}={irr}$"
        )
        plt.ylabel("degree vector")
        plt.xlabel(r"Probability difference; $P(v)-P_{TR}(v)$")
        pd.Series(diff_dict).sort_index().plot.barh()
        plt.grid(True, axis="x")
        # plt.xlim(-0.2, 0.2)
        plt.tight_layout()
        if dir_path is not None and file_name is not None:
            plt.savefig(f"{save_dir({dir_path})}/{file_name}.png")
        else:
            plt.show()
        plt.close()
