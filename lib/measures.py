from functools import reduce
from operator import add
from math import sqrt, pi, exp, log2

def l2(d1, d2):
    return sqrt(reduce(add, map(lambda y1, y2: (y1 - y2) ** 2, d1, d2)))

def kl(d1, d2):
  return reduce(add, map(lambda e1, e2: e1 * log2(e1 / e2), d1, d2))
