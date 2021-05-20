import helpers
import numpy as np


def random_sample(channel):
    rng = np.random.default_rng()
    if isinstance(channel, helpers.Reals):
        return rng.standard_gamma(1)
    if isinstance(channel, helpers.Naturals):
        return rng.geometric(0.5) - 1
    if isinstance(channel, helpers.Sequence):
        return rng.choice(range(channel.len))
    if isinstance(channel, helpers.Interval):
        return rng.uniform(0.0, channel.len)
