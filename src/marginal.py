import numpy


def density(vals):
    vals = numpy.asarray(vals)
    sort = numpy.argsort(vals)
    dest = numpy.empty(vals.size)
    distribution = numpy.arange(1, vals.size + 1) / (vals.size + 1)
    dest[sort] = distribution
    return dest
