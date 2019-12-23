import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

save_dir = "./results/test/"

def f(x, y):
    return x ** 2 + y ** 2

delta = 1
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)

# print(f(2, 3))

Z = f(X, Y)

plt.contourf(X, Y, Z, 40, cmap='RdYlBu')
plt.savefig(save_dir + "test1.png")
