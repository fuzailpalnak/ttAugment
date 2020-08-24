import numpy as np


class Arithmetic:
    def collect(self, x, y):
        raise NotImplementedError

    def aggregate(self, x, count):
        raise NotImplementedError


class Mean(Arithmetic):
    def collect(self, x, y):
        return x + y

    def aggregate(self, x, count):
        return x / count


class GeometricMean(Arithmetic):
    def collect(self, x, y):
        if not np.any(x):
            x += 1
        if not np.any(y):
            y += 1
        return x * y

    def aggregate(self, x, count):
        return x ** (1 / float(count))


def mean():
    return Mean()


def geometric_mean():
    return GeometricMean()
