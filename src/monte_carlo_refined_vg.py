import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from tools.convenient_functions import generate_ts
from tools.save import save_dir
from vg_class import RefinedSubGraph, RefinedVG

plt.rcParams["font.size"] = 15
plt.rcParams["axes.formatter.use_mathtext"] = True

script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main(omega):
    ts_kind_list = [
        "White noise",
        "Chaotic logistic map",
        "Unbiased additive random walk",
        "Additive random walk with positive drift",
        "Unbiased additive random walk with memory",
        "Unbiased multiplicative random walk",
        "Multiplicative random walk with negative drift",
        "Multiplicative random walk with volatility clustering (GARCH)",
    ]
    n_iter = 10
    max_power_idx = 17
    for i, ts_kind in enumerate(ts_kind_list, 1):
        whole_graph_list = [
            RefinedVG(
                generate_ts(kind=ts_kind, size=2 ** max_power_idx, seed=i),
                "",
                omega,
            )
            for i in range(n_iter)
        ]
        result_df = pd.DataFrame(
            {
                N: [
                    RefinedSubGraph(vg, N).compute_irreversibility()
                    for vg in whole_graph_list
                ]
                for N in [2 ** i for i in range(5, max_power_idx + 1)]
            }
        )
        result_df.to_csv(
            f"{save_dir(script_name)}/"
            f"monte_carlo_result_{ts_kind}_omega{omega}.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omega",
        type=int,
    )
    args = parser.parse_args()
    omega = args.omega
    main(omega)
