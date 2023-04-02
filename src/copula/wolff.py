from pycopcor.copula._wolff import _gamma, _sigma, _spearman


def gamma(Fx, Fy, sample_points=100):
    return _gamma(Fx, Fy, sample_points)


def sigma(Fx, Fy, sample_points=100):
    return _sigma(Fx, Fy, sample_points)


def spearman(Fx, Fy, sample_points=100):
    return _spearman(Fx, Fy, sample_points)
