import os
import re

import pandas as pd
from tools.save import save_dir
from vg_class import RefinedVG

script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main():
    market_list = ["N225", "BSESN", "HSI", "FCHI", "DJI", "GDAXI"]
    period_length = 1000
    for market in market_list:
        result = get_deg_vec_consecutive_tir(market, period_length)
        result.to_csv(f"{save_dir(script_name)}/{market}_dv-vg_tir.csv")


def get_deg_vec_consecutive_tir(market, period_length):
    market_data = (
        pd.read_csv(
            f"../data/daily_stock_prices/{market}.csv",
            usecols=["Date", "Close"],
            parse_dates=["Date"],
        )
        .dropna()
        .rename(columns={"Close": "Price"})
        .set_index("Date")
    )
    rvg = RefinedVG(
        time_series=market_data.Price.values, name=f"{market}-Refined-VG", window_width=2
    )
    tir_seq = pd.Series(
        {
            market_data.loc[i - 2 + period_length, "Date"]: rvg.compute_irreversibility(
                start=i, end=i + period_length
            )
            for i in range(1, rvg.N - period_length + 2)
        },
        name="TIR",
    )
    return market_data.join(tir_seq)


if __name__ == "__main__":
    main()
