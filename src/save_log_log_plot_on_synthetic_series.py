import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tools.save import save_dir

plt.rcParams["font.size"] = 15
plt.rcParams["axes.formatter.use_mathtext"] = True


def generate_name_mapper():
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
    name_list = [
        "(A) White noise",
        "(B) Chaotic logistic map",
        "(A) Unbiased ARW",
        "(B) ARW with positive drift",
        "(C) Unbiased ARW with memory",
        "(A) Unbiased MRW",
        "(B) MRW with negative drift",
        "(C) GARCH",
    ]
    name_mapper = {kind: name for kind, name in zip(ts_kind_list, name_list)}
    return name_mapper


def read_results(ts_kind):
    vg_data = pd.read_csv(
        f"../output/monte_carlo_omega_vg/original_vg-None_mc_result_{ts_kind}.csv",
        index_col=0,
    )
    lvg_data = pd.read_csv(
        f"../output/monte_carlo_omega_vg/original_vg-100_mc_result_{ts_kind}.csv",
        index_col=0,
    )
    dvg_data = pd.read_csv(
        f"../output/monte_carlo_refined_vg/deg-vec_vg-2_mc_result_{ts_kind}.csv",
        index_col=0,
    )
    return [vg_data, lvg_data, dvg_data]


def visualize_three_measures(ts_kind_list, name_mapper, figure_title):
    n = len(ts_kind_list)
    results_dic = {ts_kind: read_results(ts_kind) for ts_kind in ts_kind_list}
    title_list = ["VG", "LVG", "DVG"]
    color_list = ["#7249F5", "#3CA832", "#FF690D"]
    figure = plt.figure(figsize=(6.2 * n, 5))
    gs_master = GridSpec(nrows=1, ncols=n)
    plot_space = GridSpecFromSubplotSpec(nrows=1, ncols=n, subplot_spec=gs_master[0, :])

    for i, (ts_kind, result) in enumerate(results_dic.items()):
        figure.add_subplot(plot_space[:, i])
        plt.title(name_mapper[ts_kind], fontsize=22, pad=10)
        plt.ylim(1e-6, 20)
        plt.xticks(rotation=45)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Time series length: $N$")
        factor = 1
        for j, (data, title, color) in enumerate(zip(result, title_list, color_list)):
            mean = data.mean()
            eb = plt.errorbar(
                data.columns.astype(int) * factor,
                mean,
                yerr=[mean - data.quantile(0.1), data.quantile(0.9) - mean],
                c=color,
                capsize=4,
                label=title,
                lw=1.6,
                elinewidth=1.3,
            )
            eb[-1][0].set_linestyle("--")
            factor *= 1.08
        if i == 0:
            plt.ylabel("KLD")
            plt.legend(loc="lower left")
    plt.savefig(f"{save_dir('figures_for_paper')}/log-log_plot_{figure_title}.png")
    plt.close()
