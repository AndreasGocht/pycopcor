from pycopcor.copula._entropy import _empirical_2d_block, _empirical_3d_block
import numpy
import scipy.optimize


def estimate_bin_width(n, d):
    return n**(-1 / (2 + d))


def estimate_bins(n, d):
    est = estimate_bin_width(n, d)
    bins = round(1 / est)
    return bins


def norm_2d(x):
    if (x < 0):
        return numpy.nan
    return numpy.sqrt(1 - numpy.exp(-2 * x))


def norm_3d(val):
    if (val < 0):
        return numpy.nan
    d_1 = 3 - 1
    def f(p): return - 1 / 2 * numpy.log((1 - p)**d_1 * (1 + d_1 * p)) - val
    ret = scipy.optimize.root_scalar(f, bracket=(0, 1))
    if not ret.converged:
        raise RuntimeError("Not Converged")
    return ret.root


def empirical_2d(Fx, Fy, sample_points=100, offset=100):
    return _empirical_2d_block(Fx, Fy, sample_points, offset)


def empirical_3d(Fx, Fy, Fz, sample_points=100, offset=100):
    return _empirical_3d_block(Fx, Fy, Fz, sample_points, offset)


def histogram_2d(Fx, Fy, bins=None):
    if (Fx.size != Fy.size):
        raise RuntimeError("Fx and Fy must have the same size!")

    n = Fx.size
    if not bins:
        bins = estimate_bins(n, 2)

    bins_edges = numpy.linspace(0, 1, bins + 1, endpoint=True)
    h = (1 / bins)
    h_d = h ** 2

    N_d, _, _ = numpy.histogram2d(Fx, Fy, bins=bins_edges)

    N_d_log = N_d * numpy.log(N_d, where=numpy.logical_not(N_d == 0))
    N_d_sum = numpy.sum(N_d_log, axis=None, where=numpy.logical_not(numpy.isnan(N_d_log)))
    mi_h = 1 / n * N_d_sum - numpy.log(n * h_d)
    return mi_h


def histogram_3d(Fx, Fy, Fz, bins=None):
    if (Fx.size != Fy.size):
        raise RuntimeError("Fx and Fy must have the same size!")

    n = Fx.size
    if not bins:
        bins = estimate_bins(n, 3)

    bins_edges = numpy.linspace(0, 1, bins + 1, endpoint=True)
    h = (1 / bins)
    h_d = h ** 3

    F = numpy.asarray([Fx, Fy, Fz])
    F = numpy.transpose(F)

    N_d, _ = numpy.histogramdd(F, bins=(bins_edges, bins_edges, bins_edges))

    N_d_log = N_d * numpy.log(N_d, where=numpy.logical_not(N_d == 0))
    N_d_sum = numpy.sum(N_d_log, axis=None, where=numpy.logical_not(numpy.isnan(N_d_log)))
    mi_h = 1 / n * N_d_sum - numpy.log(n * h_d)
    return mi_h
