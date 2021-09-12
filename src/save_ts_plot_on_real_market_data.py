import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from tools.save import save_dir

plt.rcParams["font.size"] = 15
plt.rcParams["axes.formatter.use_mathtext"] = True


def main():
    market_list = ["N225", "BSESN", "HSI", "FCHI", "DJI", "GDAXI"]
    market_name_list = ["(A) N225", "(B) SENSEX", "(C) HSI", "(D) CAC40", "(E) DJI", "(F) DAXI"]
    calibrated_sim_result = pd.read_csv(
        "../output/fit_garch_to_real_market_data/garch_dv-vg_tir.csv", index_col=0
    ).apply(lambda data: pd.Series({"mean": data.mean(), "std": data.std()}), axis=1)
    # set figure
    figure = plt.figure(figsize=(18, 7))
    gs_master = GridSpec(nrows=2, ncols=3)
    plot_space = GridSpecFromSubplotSpec(
        nrows=2, ncols=3, subplot_spec=gs_master[:, :], wspace=0.47, hspace=0.3
    )
    for i, (market, title) in enumerate(zip(market_list, market_name_list)):
        # plot kld
        # real data
        result = (
            pd.read_csv(
                f"../output/compute_real_market_tir/{market}_dv-vg_tir.csv", parse_dates=["Date"]
            )
            .set_index("Date")
            .dropna()
        )
        ax = figure.add_subplot(plot_space[i // 3, i % 3])
        (p1,) = ax.plot(result.index, result.TIR, label="KLD of real data", c="red")
        # calibrated GARCH
        x = result.index
        m_val = calibrated_sim_result.loc[market, "mean"]
        std_val = calibrated_sim_result.loc[market, "std"]
        p2 = ax.fill_between(
            x,
            [m_val + std_val for i in range(len(x))],
            [m_val - std_val for i in range(len(x))],
            fc="tomato",
            alpha=0.3,
            label="KLD of GARCH",
        )
        # arrange the outlook
        ax.set_ylabel("KLD")
        ax.set_ylim(-0.01, 0.16)
        tkw = dict(size=5, width=1.5)
        ax.yaxis.label.set_color(p1.get_color())
        ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
        # plot price
        price = ax.twinx()
        (p3,) = price.plot(result.index, result.Price / (1e3), label="Price", c="b")
        price.set_ylabel("Price [$10^3$ yen]")
        price.yaxis.label.set_color(p3.get_color())
        price.tick_params(axis="y", colors=p3.get_color(), **tkw)
        price.set_ylim(-2.5, 42)
        # arrange the global outlook
        plt.title(title, fontsize=18)
        every_year = mdates.YearLocator()
        every_three_years = mdates.YearLocator(3)
        yearsFmt = mdates.DateFormatter("%Y")
        ax.xaxis.set_minor_locator(every_year)
        ax.xaxis.set_major_locator(every_three_years)
        ax.xaxis.set_major_formatter(yearsFmt)
        if i == 0:
            ax.legend(handles=[p1, p2, p3], loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{save_dir('figures_for_paper')}/ts_plot_on_real_market.png")


if __name__ == "__main__":
    main()
