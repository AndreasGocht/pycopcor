import pycopcor.marginal.density as density
from numpy.random import default_rng
import numpy
import time

n = 1500000
#n = 100
print("gernerating {} GB of data".format(n * 8 / 1024 / 1024 / 1024))
rng = default_rng()
vals = rng.standard_normal(n)  # 8GB
print("genereated data")

FLOP = 2 * n * n
print("GFLOP to compuite: {}".format(FLOP / 1e9))


def density_numpy(vals):
    sort = numpy.argsort(vals)
    dest = numpy.empty(vals.size)
    distribution = numpy.arange(1, vals.size + 1) / vals.size
    dest[sort] = distribution
    return dest


for i in range(10):
    begin = time.time()
    dest1 = density.density_block(vals)
    end = time.time()
    diff = end - begin
    print("block: {}s, {} GFLOP/s".format(diff, FLOP / diff / 1e9))

    begin = time.time()
    dest2 = density.density(vals)
    end = time.time()
    diff = end - begin
    print("non-block: {}s, {} GFLOP/s".format(diff, FLOP / diff / 1e9))

    begin = time.time()
    dest3 = density_numpy(vals)
    end = time.time()
    diff = end - begin
    print("numpy: {}s, {} GFLOP/s".format(diff, FLOP / diff / 1e9))

    print("comparing results")
    res1 = dest1 == dest2
    res2 = dest2 == dest3
    print(all(res1) and all(res2))
