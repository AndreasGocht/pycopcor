import pycopcor.utils
import numpy

from numpy.random import default_rng
rng = default_rng()
vals = rng.standard_normal(10000)

res = pycopcor.utils.bindings_and_min_distance(vals)

print(res)
