import numpy
import numpy.random
import time
import argparse

import pycopcor.copula.wolff
import pycopcor.marginal


parser = argparse.ArgumentParser(description='Test the perfromance of wolffs measures.')
parser.add_argument('-r', nargs=None, type=int, required=False, default=10, help='rounds')
args = parser.parse_args()

#n = 1500000
n = 2000000
print("gernerating {} GB of data".format(n * 8 / 1024 / 1024 / 1024))
rng = numpy.random.default_rng()
X = rng.standard_normal(n)  # 8GB
Y = rng.standard_normal(n)  # 8GB
Z = rng.standard_normal(n)  # 8GB

Fx = pycopcor.marginal.density(X)
Fy = pycopcor.marginal.density(Y)
Fz = pycopcor.marginal.density(Z)

samples = 100

times1 = []

for i in range(args.r):

    begin = time.time()
    res1 = pycopcor.copula.wolff.gamma(Fx, Fx)
    end = time.time()
    duration = end - begin
    times1.append(duration)
    print("myone {} s, {}".format(duration, res1))

print("median {}s, mean {}s".format(numpy.median(times1), numpy.median(times1)))
