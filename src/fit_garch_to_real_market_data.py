import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from tools.save import save_dir
from vg_class import RefinedVG

plt.rcParams["font.size"] = 15
plt.rcParams["axes.formatter.use_mathtext"] = True


script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main():
    market_list = ["N225", "BSESN", "HSI", "FCHI", "DJI", "GDAXI"]
    calibrated_params = pd.DataFrame(
        {market: fit_garch_to_real_data(market) for market in market_list}
    ).T
    calibrated_params.to_csv(f"{save_dir(script_name)}/calibrated_garch_params.csv")
    sim_result = pd.concat(
        [
            simulate_calibrated_garch_tir(market, **params)
            for market, params in calibrated_params.iterrows()
        ],
        axis=1,
    ).T
    sim_result.to_csv(f"{save_dir(script_name)}/garch_dv-vg_tir.csv")


def fit_garch_to_real_data(market):
    market_data = (
        pd.read_csv(f"../data/daily_stock_prices/{market}.csv", usecols=["Date", "Close"])
        .dropna()
        .set_index("Date")
    )
    market_data["log-return"] = np.diff(np.log(market_data["Close"].values), prepend=[np.nan])
    train_data = market_data["log-return"].values[1:] * 100
    model = arch_model(train_data, mean="Zero", vol="GARCH", p=1, q=1)
    fit_result = model.fit()
    gamma, alpha, beta = fit_result.params
    return {"gamma": gamma, "alpha": alpha, "beta": beta}


def simulate_calibrated_garch_tir(market, gamma, alpha, beta):
    size = 1000
    n_iter = 50
    result = pd.Series(
        {
            i: RefinedVG(
                time_series=regenerate_garch_ts(gamma, alpha, beta, size, seed=i), window_width=2
            ).compute_irreversibility()
            for i in range(n_iter)
        },
        name=market,
    )
    return result


def regenerate_garch_ts(gamma, alpha, beta, size, seed):
    np.random.seed(seed)
    y = [0 for _ in range(size)]
    h = [1 for _ in range(size)]
    for t in range(1, size):
        h[t] = gamma + beta * h[t - 1] + alpha * (y[t - 1] ** 2)
        y[t] = np.sqrt(h[t]) * np.random.normal(0, 1)
    y = [num / 100 for num in y]
    ts = np.cumprod(np.exp(y))
    return ts


if __name__ == "__main__":
    main()
