import numpy as np
import matplotlib.pyplot as plt
import sys

# todo: move to init
plot_dir = "/home/storm/Pictures/plots/"
figsize = (16, 8)

def plot(funs, bounds, name):
    left_bound, right_bound, step = bounds
    x = np.arange(left_bound, right_bound, step)

    fig = plt.figure(figsize=figsize)
    colors = ['g', 'b', 'r']
    for i in range(len(funs)):
        y = list(map(funs[i][0], x))
        plt.plot(x, y, colors[i % len(colors)] + 'o-', label=funs[i][1])

    # todo: move to init
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.grid(True)
    plt.legend(loc=0, fontsize=20)

    # plt.show()
    plt.savefig(plot_dir + name)
    plt.close(fig)

def compare(dist, samples,
            var, name, bins=200,
            left_bound=-10, right_bound=10, step=0.1):
    linspace = np.arange(left_bound, right_bound, step)

    fig = plt.figure(figsize=figsize)
    plt.plot(linspace,
             np.array(list(map(dist.pdf, linspace))),
             'go-', label="Target density")

    plt.hist(samples, bins=bins, density=True, label="Samples")

    plt.title("Proposal distribution variance = %.1f" % var,
              fontsize=20)

    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.grid(True)
    plt.legend(loc=0, fontsize=20)

    # plt.show()
    plt.savefig(plot_dir + name)
    plt.close(fig)

def parzen_plain(x_points, y_points,
                 title, label, x_label, y_label,
                 file_path):
    fig = plt.figure(figsize=figsize)

    plt.plot(x_points, y_points, 'bo-', label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(loc=0)

    plt.savefig(file_path)
    plt.close(fig)

def parzen_hist(dist_desc,
                dist_points, bins,
                file_path):
    fig = plt.figure(figsize=figsize)

    plt.plot(dist_points,
             [0] * len(dist_points),
             'k|',
             label="dist points")

    plt.hist(dist_points, bins=bins,
             density=True, label="point count")

    plt.title(dist_desc)

    plt.xlabel("x")
    plt.ylabel("count")
    plt.grid(True)
    plt.legend(loc=0)

    plt.savefig(file_path)
    plt.close(fig)


def parzen_compare(dist_desc, dist_points,
                   check_points, dist_dens, est_dens,
                   point_count, w_size,
                   file_path):
    fig = plt.figure(figsize=figsize)

    plt.plot(dist_points,
             [0] * len(dist_points),
             'g|',
             label="dist points")
    plt.plot(check_points,
             dist_dens,
             'go-',
             label="dist density")
    plt.stem(check_points,
             est_dens,
             '-.',
             basefmt=" ",
             label="est density")

    plt.title("Parzen window density estimation"
              + "\nDistrib: %s" % dist_desc
              + "\npoint count = %d" % point_count
              + "\nwindow size = %.2f" % w_size)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.grid(True)
    plt.legend(loc=0)

    plt.savefig(file_path)
    plt.close(fig)

def parzen_plain_2v(x, y, z,
                    title, x_label, y_label,
                    file_path):
    fig = plt.figure(figsize=figsize)

    plt.contourf(x, y, z,
                 40, cmap='RdYlBu')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(loc=0)

    plt.savefig(file_path)
    plt.close(fig)

