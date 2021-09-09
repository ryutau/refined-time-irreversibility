from collections import defaultdict

import numpy as np


def KLD(p_cnt, q_cnt, delta=1e-10):
    """
    Args:
        p_dist, q_dist (dict-like object):
            key with indegree value,
            value with # of nodes that have the key value indegree.
            e.g. {3: 2, 2: 1, 1: 5, 0: 6}
        delta (float):
            value to complement zero.
    """
    p_sum = sum(p_cnt.values())
    p_dist = {k: v / p_sum for k, v in p_cnt.items()}
    q_sum = sum(q_cnt.values())
    q_cnt = {k: v / q_sum for k, v in q_cnt.items()}
    q_dist = defaultdict(lambda: 0)
    q_dist.update(q_cnt)
    div_list = [
        p_proba * np.log((p_proba + delta) / (q_dist[k] + delta)) for k, p_proba in p_dist.items()
    ]
    kld = sum(div_list)
    return kld


def generate_ts(kind, size, seed=42):
    np.random.seed(seed)
    if kind == "White noise":
        ts = np.random.uniform(0, 1, size)
    elif kind == "Chaotic logistic map":
        x0 = np.random.uniform(0, 1)
        ts = [x0]
        for i in range(size - 1):
            x = ts[i]
            ts.append(4 * x * (1 - x))
    elif kind == "Unbiased additive random walk":
        ts = np.cumsum(np.random.uniform(low=-0.5, high=0.5, size=size))
    elif kind == "Additive random walk with positive drift":
        np.random.seed(seed + 100)
        ts = np.cumsum(np.random.uniform(low=-0.4, high=0.6, size=size))
    elif kind == "Unbiased additive random walk with memory":
        r = 0.3
        tau = 6
        ts = np.cumsum(np.random.uniform(-0.5, 0.5, tau))
        for i in range(tau, size):
            p = np.random.uniform()
            if p > r:
                x_before = ts[i - 1]
                xi = np.random.uniform(-0.5, 0.5)
                ts = np.append(ts, x_before + xi)
            else:
                memory = ts[i - tau]
                ts = np.append(ts, memory)
    elif kind == "Unbiased multiplicative random walk":
        ts = np.cumprod(np.append([1], np.exp(np.random.uniform(-0.5, 0.5, size - 1))))
    elif kind == "Multiplicative random walk with negative drift":
        ts = np.cumprod(np.append([1], np.random.uniform(0.9, 1.1, size - 1)))
    elif kind == "Multiplicative random walk with volatility clustering (GARCH)":
        gamma = 0.1
        # gamma = 0.045
        beta = 0.6
        # beta = 0.87
        alpha = 0.3
        # alpha = 0.11
        y = [0 for _ in range(size)]
        h = [1 for _ in range(size)]
        for t in range(1, size):
            h[t] = gamma + beta * h[t - 1] + alpha * (y[t - 1] ** 2)
            y[t] = np.sqrt(h[t]) * np.random.normal(0, 1)
        y = [num / 100 for num in y]
        ts = np.cumprod(np.exp(y))
    else:
        error_message = (
            "Arg 'kind' must be one of the folloing.\n'White noise',\n"
            "'Chaotic logistic map',\n'Unbiased additive random walk',\n"
            "'Additive random walk with positive drift',\n"
            "'Unbiased additive random walk with memory',\n"
            "'Unbiased multiplicative random walk',\n"
            "'Multiplicative random walk with negative drift' ,\n."
            "'GARCH'."
        )
        raise ValueError(error_message)
    return ts


def nested_dict():
    return defaultdict(nested_dict)
