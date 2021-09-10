import argparse
import os
import re
import time
from collections import defaultdict

import pandas as pd
from tools.convenient_functions import generate_ts
from tools.save import save_dir
from vg_class import BasicSubGraph, EfficientVisibilityGraph

script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main(omega):
    start = time.time()
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
    max_power_idx = 16  # max size of the time series will be 2**max_power_idx
    for ts_kind in ts_kind_list:
        result_dict = defaultdict(list)
        for i in range(n_iter):
            print(i)
            vg = EfficientVisibilityGraph(
                generate_ts(kind=ts_kind, size=2 ** max_power_idx, seed=i),
                name="",
                window_width=omega,
            )
            for N in [2 ** j for j in range(5, max_power_idx + 1)]:
                result_dict[N].append(BasicSubGraph(vg, N).compute_irreversibility())
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(f"{save_dir(script_name)}/vg-{omega}_mc_result_{ts_kind}.csv")
        print(f"{ts_kind} has been finished.")
    elapsed_time = time.time() - start
    print(f"elapsed_time:{elapsed_time}[sec]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omega",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    omega = args.omega
    main(omega)
