from scipy.interpolate import SmoothBivariateSpline, RectBivariateSpline
from scipy.stats import multivariate_normal
import numpy as np


def gen_cubic_noise_functional(w_min, w_tot, persistence, seed=None):
    if seed:
        np.random.seed(seed)
    norm_factor = 0
    splines = []
    factors = []
    for w in range(w_min, w_min + w_tot):
        rand = np.random.uniform(size=(2**w, 2**w))
        net_points = np.linspace(0, 1, 2**w, endpoint=True)
        net_points = np.linspace(0, 1, 2**w, endpoint=True)
        spline = RectBivariateSpline(x=net_points, y=net_points, z=rand, kx=3, ky=3)
        factor = persistence**w
        splines.append(spline)
        factors.append(factor)
        norm_factor += factor

    def func(x, y):
        depth = np.sum(s(x, y, grid=False) * f for f, s in zip(factors, splines)) / norm_factor
        return depth
    return func
