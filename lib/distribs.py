import numpy as np
from math import sqrt, pi, exp, log2
from scipy.special import gamma


class Normal:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def set_mean(self, mean):
        self.mean = mean

    def set_var(self, var):
        self.var = var

    # probability density function
    def pdf(self, x):
        return 1 / sqrt(2 * pi * self.var ** 2) \
               * exp(- (x - self.mean) ** 2 / (2 * self.var ** 2))

    # random variates
    def rvs(self, count=1):
        v = np.random.normal(self.mean, self.var, int(count))
        if count == 1:
            return v[0]
        return v

    def name(self):
        return "normal"

    def desc(self):
        return "Normal(%.2f, %.2f)" % (self.mean, self.var)

class Uniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        return 0

    def rvs(self, count=1):
        v = np.random.uniform(self.a, self.b, int(count))
        if count == 1:
            return v[0]
        return v

    def name(self):
        return "uniform"

    def desc(self):
        return "Uniform(%.2f, %.2f)" % (self.a, self.b)

class Beta:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        return gamma(self.a + self.b) / (gamma(self.a) * gamma(self.b)) \
                 * x ** (self.a - 1) * (1 - x) ** (self.b - 1)

    def rvs(self, count=1):
        v = np.random.beta(self.a, self.b, int(count))
        if count == 1:
            return v[0]
        return v

    def name(self):
        return "beta"

    def desc(self):
        return "Beta(%.2f, %.2f)" % (self.a, self.b)

class NormalMixture:
    def __init__(self, mu):
        self.mu = mu

    def pdf(self, x):
      return 1 / (2 * sqrt(2 * pi)) * (exp(-(x - self.mu) ** 2 / 2)
             + exp(-(x + self.mu) ** 2 / 2))
