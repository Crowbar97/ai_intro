import numpy as np
from ai_intro.lib.distribs import Distribs
from ai_intro.lib.plotter import Plotter

class MHSampler:
    @staticmethod
    def metropolis_ratio(target_dist, x0, x1):
        return target_dist.pdf(x1) / target_dist.pdf(x0)

    @staticmethod
    def hastings_ratio(prop_dist, x0, x1):
        prop_dist.set_mean(x0)
        q0 = prop_dist.pdf(x1)

        prop_dist.set_mean(x1)
        q1 = prop_dist.pdf(x0)

        prop_dist.set_mean(x0)

        return q1 / q0

    @staticmethod
    def accepted(accept_prob):
        return np.random.binomial(1, min(accept_prob, 1), 1)[0]

    @staticmethod
    def discard(samples, disc_part):
        del samples[:int(len(samples) * disc_part)]

    @staticmethod
    def sample(target_dist, prop_dist, x_init=0,
               sample_count=1e5, disc_part=0.1):
        x0 = x_init
        target_samples = [x0]
        sample_count -= 1
        while(sample_count):
            prop_dist.set_mean(x0)

            x1 = prop_dist.rvs()

            accept_prob = MHSampler.metropolis_ratio(target_dist, x0, x1) \
                          * MHSampler.hastings_ratio(prop_dist, x0, x1)

            if MHSampler.accepted(accept_prob):
                target_samples.append(x0)
                x0 = x1
                sample_count -= 1

        MHSampler.discard(target_samples, disc_part)
        prop_dist.set_mean(x_init)

        return target_samples

