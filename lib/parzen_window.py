def rect_kernel(z):
    return 0.5 * int(abs(z) < 1)

def density(x, points, d):
    sum = 0
    for p in points:
        sum += rect_kernel((x - p) / d)
    return 1 / (len(points) * d) * sum

def estimate_density(distr_points, d, test_points):
    return list(map(lambda p:
                        density(p, distr_points, d),
                    test_points))
