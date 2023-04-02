import argparse
import numpy
import numpy.random
import scipy.optimize

from pycopcor import marginal
from pycopcor.copula import entropy
from pycopcor.copula import wolff


def gen_data(corr, d, n):
    cov = numpy.full(shape=(d, d), fill_value=corr)
    numpy.fill_diagonal(cov, 1)

    mean = numpy.zeros((d))
    data = numpy.random.multivariate_normal(mean=mean, cov=cov, size=n)
    return data


parser = argparse.ArgumentParser(description='Test the different copula besed correlation measures.')
parser.add_argument('-d', nargs=None, type=int, choices=(2, 3), required=True, help='2d or 3d to compute')
parser.add_argument('-n', nargs=None, type=int, default=10000, required=False, help='Elements to compute')
parser.add_argument('-c', '--correlation', nargs=None, type=float, default=0.5,
                    required=False, help='correlation of equicorrelated gaussian')
parser.add_argument('-o', '--offset', nargs=None, type=float, default=100,
                    required=False, help='offset for the empircial copulas')
parser.add_argument('-s', '--sample-points', nargs=None, type=int, default=100,
                    required=False, help='sample points for the empircial copulas')
parser.add_argument('-b', '--bins', nargs=None, type=int, default=None,
                    required=False, help='bins for the historical copulas')
args = parser.parse_args()

d = args.d
n = args.n
corr = args.correlation
offset = args.offset
sample_points = args.sample_points
bins = args.bins

if not bins:
    bins = entropy.estimate_bins(n, d)

data = gen_data(corr, d, n)

Fx = marginal.density(data[:, 0])
Fy = marginal.density(data[:, 1])
if d == 3:
    Fz = marginal.density(data[:, 2])

if d == 2:

    norm_max = entropy.histogram_2d(Fx, Fx)
    res = entropy.histogram_2d(Fx, Fy, bins=bins)
    print("histogram_2d ({:>2} bins):     {:.5f} , {:.5f} , {:.5f}".format(
        bins, res, entropy.norm_2d(res), res / norm_max))

    norm_max = entropy.empirical_2d(Fx, Fx, sample_points=sample_points, offset=offset)
    res = entropy.empirical_2d(Fx, Fy, sample_points=sample_points, offset=offset)
    print("empirical_2d:               {:.5f} , {:.5f} , {:.5f}".format(res, entropy.norm_2d(res), res / norm_max))

    res = wolff.gamma(Fx, Fy)
    print("gamma:              {}".format(res))

    res = wolff.sigma(Fx, Fy)
    print("sigma:              {}".format(res))

    res = wolff.spearman(Fx, Fy)
    print("spearman:           {}".format(res))

elif d == 3:
    norm_max = entropy.histogram_3d(Fx, Fx, Fx, bins=bins)
    res = entropy.histogram_3d(Fx, Fy, Fz, bins=bins)
    print("histogram_3d ({:>2} bins): {:.5f} , {:.5f} , {:.5f}".format(bins, res, entropy.norm_3d(res), res / norm_max))

    norm_max = entropy.empirical_3d(Fx, Fx, Fx, sample_points=sample_points, offset=offset)
    res = entropy.empirical_3d(Fx, Fy, Fz, sample_points=sample_points, offset=offset)
    print("empirical_3d:           {:.5f} , {:.5f} , {:.5f}".format(res, entropy.norm_3d(res), res / norm_max))


# Fx = marginal.density(X)
# Fy = marginal.density(Y)
