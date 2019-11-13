import numpy as np
from ai_intro.lib.distribs import Normal, Uniform, Beta
from ai_intro.run.plotter import parzen_hist, parzen_compare, parzen_plain, parzen_compare
from ai_intro.lib.parzen_window import estimate_density
from ai_intro.lib.measures import l2

save_dir = "/home/storm/Pictures/ai_intro/"

# Explores dist sampling
def explore_sampling(dist, point_count):
    print("Exploring \"%s\" sampling..." % dist.name())
    dist_points = dist.rvs(point_count)

    parzen_hist(dist.desc() + " sampling: %.0e points" % point_count,
                dist_points, 200,
                save_dir, dist.name() + "_points.png")

# Explores how parzen window density estimation
# quality depends on sample point count (by l2-distance)
def explore_point_count(dist, check_points,
                        w_size, point_counts):
    print("Exploring point count for \"%s\" dist..." % dist.name())
    l2s = []
    l2_min = 1e9
    best_pc = point_counts[0]
    dist_dens = list(map(lambda x:
                            dist.pdf(x),
                         check_points))
    # FIXME: estimate mean value of est. density
    for point_count in point_counts:
        print("Est-ing density for %d points..." % point_count)
        dist_points = dist.rvs(point_count)
        est_dens = estimate_density(dist_points, w_size, check_points)
        l2d = l2(dist_dens, est_dens)
        if l2d < l2_min:
            l2_min = l2d
            best_pc = point_count
        l2s.append(l2d)

    parzen_plain(point_counts, l2s,
          "Parzen window (size = %.2f)" % w_size
          + " estimation quality by L2 distance"
          + "\nfor " + dist.desc(),
          "L2", "sample point count", "L2",
          save_dir, dist.name() + "_l2_of_count.png")

    return best_pc


# Explores how parzen window density estimation
# quality depends on window size (by l2-distance)
def explore_window_size(dist, check_points,
                        w_sizes, point_count):
    print("Exploring window size for \"%s\" dist..." % dist.name())
    l2s = []
    l2_min = 1e9
    best_ws = w_sizes[0]
    dist_dens = list(map(lambda x:
                            dist.pdf(x),
                         check_points))
    for w_size in w_sizes:
        print("Est-ing density for window size = %.2f..." % w_size)
        dist_points = dist.rvs(point_count)
        est_dens = estimate_density(dist_points, w_size, check_points)
        l2d = l2(dist_dens, est_dens)
        if l2d < l2_min:
            l2_min = l2d
            best_ws = w_size
        l2s.append(l2d)

    parzen_plain(w_sizes, l2s,
          "Parzen window (sample point count = %.0e)" % point_count
          + " estimation quality by L2 distance"
          + "\nfor " + dist.desc(),
          "L2", "window size", "L2",
          save_dir, dist.name() + "_l2_of_size.png")

    return best_ws


def explore_best_params(dist, check_points,
                        best_pc, best_ws):
    print("Est-ing density for best params:"
          + "\npoint count = %.0e, window size = %.2f"
          % (best_pc, best_ws))

    dist_points = dist.rvs(best_pc)
    est_dens = estimate_density(dist_points, best_ws, check_points)
    dist_dens = list(map(lambda x:
                            dist.pdf(x),
                         check_points))

    parzen_compare(dist.desc(), dist_points,
                   check_points, dist_dens, est_dens,
                   best_pc, best_ws,
                   save_dir, dist.name() + "_best.png")


# Check list
def explore(dist, check_points):
    point_count = int(1e4)
    explore_sampling(dist, point_count)

    w_size = 0.1
    point_counts = range(int(1e3), int(1e4) + 1, int(5e2))
    best_pc = explore_point_count(dist, check_points,
                                  w_size, point_counts)

    point_count = int(1e4)
    w_sizes = np.arange(0.01, 0.1, 0.01)
    best_ws = explore_window_size(dist, check_points,
                                  w_sizes, point_count)

    explore_best_params(dist, check_points,
                        best_pc, best_ws)


# -Normal dist-
def explore_normal():
    mean = 0
    var = 1
    norm_dist = Normal(mean, var)
    check_points = np.arange(-4, 4, 0.1)
    explore(norm_dist, check_points)


# -Uniform dist-
def explore_uniform():
    a = -2
    b = 2
    unif_dist = Uniform(a, b)
    check_points = np.arange(-4, 4, 0.1)
    explore(unif_dist, check_points)

# -Beta dist-
def explore_beta():
    a = 0.5
    b = 0.5
    check_points = np.arange(0.01, 1, 0.01)
    beta_dist = Beta(a, b)
    explore(beta_dist, check_points)

explore_normal()
explore_uniform()
explore_beta()

# TODO: optimize?
# TODO: make l2(counts, sizes)? - for Beta dist

# parzen_compare(norm_dist.desc(), norm_points,
#                check_points, norm_dens, est_dens,
#                point_count, w_size,
#                save_dir, "comp.png")
