from abc import ABCMeta, abstractmethod
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from tools.convenient_functions import KLD

plt.rcParams["font.size"] = 15


class BaseGraph(metaclass=ABCMeta):
    def __init__(self, time_series, name):
        """
        Args:
            time_series (array-like object):
                The original time series.
            kind (str):
                Should either be 'visible' or 'invisible'.
        """
        self.name = name
        self.N = len(time_series)
        self.ts = [(t, x) for t, x in enumerate(time_series, 1)]
        self.graph = self.make_graph()

    @abstractmethod
    def make_graph(self):
        pass

    def get_degree_sequence(self, deg_kind):
        if deg_kind == "degree":
            degree_sequence = {n: d for n, d in self.graph.degree()}
        elif deg_kind == "indegree":
            degree_sequence = {n: d for n, d in self.graph.in_degree()}
        elif deg_kind == "outdegree":
            degree_sequence = {n: d for n, d in self.graph.out_degree()}
        else:
            raise ValueError(
                "deg_kind should either be degree, indegree or outdegree"
            )
        return degree_sequence

    def get_degree_cnt_dict(self, **kargs):
        return Counter(self.get_degree_sequence(**kargs).values())

    def compute_irreversibility(self):
        indegree_dict = self.get_degree_cnt_dict(deg_kind="indegree")
        outdegree_dict = self.get_degree_cnt_dict(deg_kind="outdegree")
        kld = KLD(indegree_dict, outdegree_dict)
        return kld

    def plot_original_time_series(self, tstart=1, tend=None, save_path=None):
        if tend is None:
            tend = self.N
        plt.figure(figsize=(12, 6))
        ts_dic = {data[0]: data[1] for data in self.ts}
        pd.Series(ts_dic).loc[tstart:tend].plot()
        plt.suptitle(f"{self.name}", fontsize=20)
        plt.title(f"Original Time Series(trange: {tstart}~{tend})", fontsize=15)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        if save_path is not None:
            plt.savefig(f"{save_path}.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_network(self, tstart=1, tend=None, save=False, save_path=None):
        if tend is None:
            tend = self.N
        g = self.graph.subgraph(range(tstart, tend + 1))
        fig, ax = plt.subplots(figsize=(12, 6))
        pos = {i: (i, g.nodes[i]["value"]) for i in range(tstart, tend + 1)}
        nx.draw_networkx(
            g,
            pos=pos,
            with_labels=False,
            node_size=30,
            node_color="black",
            ax=ax,
            width=0.7,
        )
        plt.plot(
            [n[0] for n in self.ts[tstart - 1 : tend]],
            [n[1] for n in self.ts[tstart - 1 : tend]],
            c="k",
            mfc="lightgreen",
            ms=20,
            marker=".",
        )
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        xticks = [i for i in range(tstart, tend + 1) if i % 5 == 0]
        plt.suptitle(f"{self.name}", fontsize=20)
        plt.title(f"Network Structure(trange: {tstart}~{tend})", fontsize=15)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.xticks(xticks)
        if save:
            plt.savefig(f"{save_path}.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_degree_histgram(
        self, deg_kind="degree", save=False, save_path=None, **kwargs
    ):
        deg, cnt = zip(*self.get_degree_cnt_dict(deg_kind=deg_kind).items())
        cnt /= np.sum(cnt)
        plt.figure(figsize=(12, 6))
        plt.bar(deg, cnt, width=0.80, **kwargs)
        plt.suptitle(f"{self.name}", fontsize=20)
        plt.title(f"Histgram of nodes' {deg_kind}s", fontsize=15)
        plt.xlabel("Degree")
        plt.ylabel("Ratio")
        if save:
            plt.savefig(f"{save_path}.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def log_log_index_plot(self, save=False, save_path=None):
        indegree_sequence = sorted(
            [d for n, d in self.graph.in_degree()], reverse=True
        )
        d_list = np.unique(indegree_sequence)
        p_list = [
            len([dd for dd in indegree_sequence if dd >= d])
            / len(indegree_sequence)
            for d in d_list
        ]
        i_list = d_list / self.window_width
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(i_list, p_list, "*")
        ax.grid(True)
        fig.suptitle(f"{self.name}", fontsize=20)
        ax.set_title(
            "Cumulative step histograms of Indegree Index", fontsize=15
        )
        ax.set_xlabel("Indegree Index")
        ax.set_ylabel("Cumulative probability (P(I >= x))")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1e0)
        ax.set_ylim(1e-4, 1e0)
        if save:
            plt.savefig(f"{save_path}.png", bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()
        plt.close()
