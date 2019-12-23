import numpy as np
from lib.distribs import Normal
from run.plotter import parzen_plain
from lib.measures import kl
from lib.parzen_window import estimate_density

save_dir = "./results/kl/"

# TODO: Use best win sizes for each point count (from parzen exploring)
# Explores how KL divergence depends on sample point count
def explore_point_count(dist, check_points,
                        w_size, point_counts,
                        repeat_count):
    print("Exploring point count for \"%s\" dist..." % dist.name())
    kls = []
    kl_min = 1e9
    best_pc = point_counts[0]
    dist_dens = list(map(lambda x:
                            dist.pdf(x),
                         check_points))
    for point_count in point_counts:
        print("Est-ing density for %d points..." % point_count)

        kl_avg = 0
        for _ in range(repeat_count):
            dist_points = dist.rvs(point_count)
            est_dens = estimate_density(dist_points,
                                        w_size,
                                        check_points)
            kl_avg += kl(dist_dens, est_dens)
        kl_avg /= repeat_count

        if kl_avg < kl_min:
            kl_min = kl_avg
            best_pc = point_count
        kls.append(kl_avg)

    parzen_plain(point_counts, kls,
          "Parzen window (size = %.2f)" % w_size
          + " estimation quality by KL distance"
          + "\nfor " + dist.desc(),
          "KL", "sample point count", "KL",
          save_dir + dist.name() + "_kl_of_count.png")

    return best_pc


# conditions
repeat_count = 20
point_counts = range(int(1e3), int(1e4), int(3e2))
check_points = np.arange(-3, 3, 0.1)
w_size = 0.8

# normal distribution
mean, var = 0, 1
norm_dist = Normal(mean, var)

explore_point_count(norm_dist, check_points,
                    w_size, point_counts,
                    repeat_count)

