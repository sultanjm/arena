from collections import defaultdict

from . import core
from . import helpers

import numpy as np


class GreedyQAgent(core.Actor):

    def setup(self, *args, **kwargs):
        self.g = kwargs.get('discount_factor', 0.999)
        def_val = kwargs.get('initial_value', None)
        # optimistic initialization
        self.Q = defaultdict(lambda: def_val) if def_val else defaultdict(
            lambda: 1/(1-self.g))
        self.eps = kwargs.get('exploration_factor', 0.999)

    def act(self, history, controls):
        return super().act(history, controls)

    def learn(self, history, evaluations):
        return super().learn(history, evaluations)

    def state_func(self, history, controls, actions):
        return super().state_func(history, controls, actions)
