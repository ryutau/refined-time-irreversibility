import numpy as np

from vg_class import RefinedVG


def tti(ts, omega):
    refined_vg = RefinedVG(time_series=ts, window_width=omega)
    return refined_vg.compute_irreversibility()


class ShuffleMC:
    def __init__(self, original, omega, n_iter=10):
        self.original = original
        self.omega = omega
        self.n_iter = n_iter
        np.random.seed(0)
        self.result = [
            tti(np.random.permutation(self.original), omega=self.omega)
            for _ in range(self.n_iter)
        ]
        self.mean = np.mean(self.result)
        self.std = np.std(self.result)


class SampleMC:
    def __init__(self, original, window, omega, n_iter=10):
        self.original = original
        self.window = window
        self.omega = omega
        self.n_iter = n_iter
        np.random.seed(0)
        self.result = [self.ret_irr() for _ in range(self.n_iter)]
        self.mean = np.mean(self.result)
        self.std = np.std(self.result)

    def ret_irr(self):
        sample_start = np.random.randint(1, len(self.original) - self.window)
        ts = self.original[sample_start : sample_start + self.window]
        return tti(ts, self.omega)
