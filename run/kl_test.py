import numpy as np
from ai_intro.lib.distribs import Normal
from ai_intro.lib.plotter import plain
from ai_intro.lib.measures import kl
from ai_intro.density_estimation.parzen_window import estimate_density

# conditions
test_points = np.arange(-3, 3, 0.1)
window_size = 0.4

# normal distribution
mean, var = 0, 1
norm_dist = Normal(mean, var)
norm_dens = list(map(lambda point:
                         norm_dist.pdf(point),
                     test_points))

point_counts = range(int(1e3), int(5e3), int(1e3))
kls = []
for point_count in point_counts:
    print("Estimating KL for %d  points..." % point_count)

    norm_points = norm_dist.rvs(point_count)
    est_dens = estimate_density(norm_points,
                                       window_size,
                                       test_points)
    kl_val = kl(norm_dens, est_dens)

    print("KL =", kl_val)
    kls.append(kl_val)

# Plotter.plot("normal", test_points, normal_density, est_density, point_count)
# print(Distances.kl(normal_density, est_density))
plain(point_counts, kls, "kl_div")
