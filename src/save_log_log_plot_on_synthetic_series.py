import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from tools.save import save_dir

matplotlib.rcParams.update({"figure.autolayout": True})
plt.rcParams["font.size"] = 13
plt.rcParams["axes.formatter.use_mathtext"] = True


def main():
    for ts_category in ["Stationary", "ARW", "MRW"]:
        vis_tir_loglog_plot(ts_category)


def generate_ts_name_dict(tgt_category):
    ts_kind_list = [
        ("White noise", "Stationary"),
        ("Chaotic logistic map", "Stationary"),
        ("Unbiased additive random walk", "ARW"),
        ("Additive random walk with positive drift", "ARW"),
        ("Unbiased additive random walk with memory", "ARW"),
        ("Unbiased multiplicative random walk", "MRW"),
        ("Multiplicative random walk with negative drift", "MRW"),
        ("Multiplicative random walk with volatility clustering (GARCH)", "MRW"),
    ]
    ts_title_list = [
        "(A) White noise",
        "(B) Chaotic logistic map",
        "(A) Unbiased ARW",
        "(B) ARW with positive drift",
        "(C) Unbiased ARW with memory",
        "(A) Unbiased MRW",
        "(B) MRW with negative drift",
        "(C) GARCH",
    ]
    ts_name_dict = {
        kind: title
        for (kind, category), title in zip(ts_kind_list, ts_title_list)
        if category == tgt_category
    }
    return ts_name_dict


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


def vis_tir_loglog_plot(ts_category):
    ts_name_dict = generate_ts_name_dict(ts_category)
    n = len(ts_name_dict)
    results_dict = {ts_kind: read_results(ts_kind) for ts_kind in ts_name_dict.keys()}
    label_list = ["VG", "LVG", "DVG"]
    color_list = ["#7249F5", "#3CA832", "#FF690D"]
    ls_list = [":", "-.", "--"]
    figure = plt.figure(figsize=(6.2 * n, 5))
    gs_master = GridSpec(nrows=1, ncols=n)
    plot_space = GridSpecFromSubplotSpec(
        nrows=1, ncols=n, subplot_spec=gs_master[0, :], wspace=0.15
    )
    for i, (ts_kind, result) in enumerate(results_dict.items()):
        figure.add_subplot(plot_space[:, i])
        plt.title(ts_name_dict[ts_kind], fontsize=22, pad=10)
        plt.ylim(1e-6, 20)
        plt.xticks()
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Time series length: $N$", fontsize=15)
        factor = 1
        for j, (data, label, color, ls) in enumerate(zip(result, label_list, color_list, ls_list)):
            mean = data.mean()
            eb = plt.errorbar(
                data.columns.astype(int) * factor,
                mean,
                yerr=[mean - data.quantile(0.1), data.quantile(0.9) - mean],
                c=color,
                capsize=4,
                label=label,
                lw=1.6,
                elinewidth=1.3,
            )
            eb[-1][0].set_linestyle(ls)
            factor *= 1.06
        if i == 0:
            plt.ylabel("KLD", fontsize=15)
            plt.legend(loc="lower left", fontsize=15)
    plt.savefig(f"{save_dir('figures_for_paper')}/log-log_plot_{ts_category}.png")
    plt.close()


if __name__ == "__main__":
    main()
