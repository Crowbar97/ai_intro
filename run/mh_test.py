from ai_intro.lib.distribs import Distribs
from ai_intro.sampling.metropolis_hastings import MHSampler

def main():
    # target distribution
    target_dist = Distribs.NormalMixture(1.8)

    # proposal (approximating) distribution
    var = 0.2
    prop_dist = Distribs.Normal(0, var)
    Plotter.plot([(target_dist.pdf,
                   "Target dist density"),
                  (prop_dist.pdf,
                   "Proposal dist density\n(normal, variance = %.1f)" % var)],
                 (-10, 10, 0.1),
                 "prop1")

    var = 2.0
    prop_dist = Distribs.Normal(0, var)
    Plotter.plot([(target_dist.pdf, "Target dist density"),
                  (prop_dist.pdf, "Proposal dist density\n(normal, variance = %.1f)" % var)],
                 (-10, 10, 0.1),
                 "prop2")

    for var in np.arange(0.2, 2.2, 0.2):
        print("Sampling for var = %f..." % var)
        prop_dist = Distribs.Normal(0, var)
        target_samples = MHSampler.sample(target_dist, prop_dist, x_init=0,
                                          sample_count=1e5, disc_part=0.1)
        Plotter.compare(target_dist, target_samples, var=var, name="comp" + str(int(var * 10)))

main()
