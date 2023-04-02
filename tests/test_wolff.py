import pycopcor.copula.wolff as wolff
import pycopcor.marginal as marginal
import numpy
import numpy.random

n = 10000


def test_gamma_dependence():
    X = numpy.sin(numpy.linspace(0, 2 * numpy.pi, n))
    Y = numpy.sin(numpy.linspace(0, 2 * numpy.pi, n))

    Fx = marginal.density(X)
    Fy = marginal.density(Y)

    res = wolff.gamma(Fx, Fy)
    res_ass = abs(1.0 - res)
    assert 1e-3 > res_ass


def test_gamma_independence():
    rng = numpy.random.default_rng()
    X = rng.standard_normal(n)  # 8GB
    Y = rng.standard_normal(n)  # 8GB

    Fx = marginal.density(X)
    Fy = marginal.density(Y)

    res = wolff.gamma(Fx, Fy)
    assert 1e-1 > res


def test_sigma_dependence():
    X = numpy.sin(numpy.linspace(0, 2 * numpy.pi, n))
    Y = numpy.sin(numpy.linspace(0, 2 * numpy.pi, n))

    Fx = marginal.density(X)
    Fy = marginal.density(Y)

    res = wolff.sigma(Fx, Fy)
    res_ass = abs(1.0 - res)
    assert 1e-3 > res_ass


def test_sigma_independence():
    rng = numpy.random.default_rng()
    X = rng.standard_normal(n)  # 8GB
    Y = rng.standard_normal(n)  # 8GB

    Fx = marginal.density(X)
    Fy = marginal.density(Y)

    res = wolff.sigma(Fx, Fy)
    assert 1e-1 > res


def test_spearman_dependence():
    X = numpy.sin(numpy.linspace(0, 2 * numpy.pi, n))
    Y = numpy.sin(numpy.linspace(0, 2 * numpy.pi, n))

    Fx = marginal.density(X)
    Fy = marginal.density(Y)

    res = wolff.spearman(Fx, Fy)
    res_ass = abs(1.0 - res)
    assert 1e-3 > res_ass


def test_spearman_independence():
    rng = numpy.random.default_rng()
    X = rng.standard_normal(n)  # 8GB
    Y = rng.standard_normal(n)  # 8GB

    Fx = marginal.density(X)
    Fy = marginal.density(Y)

    res = abs(wolff.spearman(Fx, Fy))
    assert 1e-1 > res
