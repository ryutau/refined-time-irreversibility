import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def indegree_idx_log_log_plot(g_list, save=False, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    c_list = ['lime', 'orange', 'green', 'salmon']
    for i, g in enumerate(g_list):
        degree_sequence =\
            sorted([d for n, d in g.graph.in_degree()], reverse=True)
        d_list = np.unique(degree_sequence)
        p_list = [len([dd for dd in degree_sequence
                       if dd >= d]) / len(degree_sequence)
                  for d in d_list]
        i_list = d_list / g.window_width
        ax.plot(i_list, p_list, '*', label=g.name, color=c_list[i])
    fig.suptitle(f'Cumulative Indegree Idx plot', fontsize=20)
    ax.grid(True)
    ax.set_xlabel('Indegree Idx')
    ax.set_ylabel('Cumulative probability (P(I >= x))')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-3, 1e0)
    ax.set_ylim(1e-4, 1e0)
    plt.legend()
    if save:
        plt.savefig(f'{save_path}.png', bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def indegree_histgram(g_list, save=False, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    c_list = ['lime', 'orange', 'green', 'salmon']
    for i, g in enumerate(g_list):
        degree_sequence =\
            sorted([d for n, d in g.graph.in_degree()], reverse=True)
        ax = fig.add_subplot(len(g_list), 1, (i+1))
        ax.hist(degree_sequence, bins=range(100),
                log=True, label=g.name, color=c_list[i])
        ax.legend()
        ax.set_ylim(1e0, 1e4)
    plt.xlabel('Indegree')
    fig.suptitle(f'Indegree Histgram', fontsize=20)
    fig.text(0.04, 0.5, '# of nodes in log-scale', va='center', rotation='vertical')
    if save:
        plt.savefig(f'{save_path}.png', bbox_inches="tight")
    else:
        plt.show()
    plt.close()
