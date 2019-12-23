from functools import reduce
from operator import add
from math import sqrt, pi, exp, log2

def l2(d1, d2):
    return sqrt(reduce(add,
                    map(lambda y1, y2:
                            (y1 - y2) ** 2,
                        d1, d2)))

def kl(d1, d2):
    eps = 1e-9
    d1 = list(map(lambda d:
                d + eps if d == 0 else d,
             d1))
    d2 = list(map(lambda d:
                d + eps if d == 0 else d,
             d2))
    return reduce(add,
                    map(lambda e1, e2:
                            e1 * log2(e1 / e2),
                        d1, d2))
