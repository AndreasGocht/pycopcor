import numpy
import numpy.random
import time

import pycopcor.copula.entropy
import pycopcor.marginal


#n = 1500000
n = 100000
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
times2 = []

for i in range(10):
    # begin = time.time()
    # res = cd3(Fx,Fy,Fz,samples, 100)
    # end = time.time()
    # duration = end-begin
    # times.append(duration)
    # print("numpy {} s, {}".format(duration,res))

    begin = time.time()
    res1 = pycopcor.copula.entropy.empirical_2d(Fx, Fx, samples, 100)
    end = time.time()
    duration = end - begin
    times1.append(duration)
    print("myone {} s, {}".format(duration, res1))

    begin = time.time()
    res2 = pycopcor.copula.entropy.empirical_2d_block(Fx, Fx, samples, 100)
    end = time.time()
    duration = end - begin
    times2.append(duration)
    print("myone_block {} s, {}".format(duration, res2))
    print(res1 - res2)

print("median {}s, mean {}s".format(numpy.median(times1), numpy.median(times1)))
print("median {}s, mean {}s".format(numpy.median(times2), numpy.median(times2)))
