from collections import defaultdict

from numpy.lib.shape_base import hsplit
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


class POMDP(core.Actor):

    def setup(self, *args, **kwargs):
        # setting up basic attributes of an MDP
        # any non-zero value is sufficient for an ergodic MDP
        self.min_probability = kwargs.get('min_probability', 1e-9)
        # control and feedback spaces are "action" and "state" spaces
        self.control_space = [helpers.Sequence(range(2))] if not kwargs.get(
            'control_space', None) else kwargs.get('control_space', None)
        self.state_space = [helpers.Sequence(range(2))] if not kwargs.get(
            'state_space', None) else kwargs.get('state_space', None)
        # state-space is internal to the actor
        # outside world should not worry about this space
        # it's the actor who knows what it is and how to manipulate it
        # it should be specified and *must* be different from ``self.state_space``
        self.feedback_space = kwargs.get('feedback_space', self.state_space)
        # random number generator
        self.rng = np.random.default_rng()
        # self.transition_matrix = kwargs.get('transition_matrix', None)
        # if not self.transition_matrix:
        #     self.transition_matrix = self.rng.random([
        #         self.num_states, self.num_actions, self.num_states]) + self.min_probability  # [s][a][s_nxt]
        #     # normalize to get a stochastic matrix
        #     self.transition_matrix = self.transition_matrix / \
        #         self.transition_matrix.sum(axis=-1, keepdims=True)
        # self.reward_matrix = kwargs.get('reward_matrix', None)
        # if not self.reward_matrix:
        #     self.reward_matrix = self.rng.random(
        #         [self.num_states, self.num_actions, self.num_states])  # [s][a][s_nxt]

    def evaluate(self, history, controls, actions, feedbacks):
        next_state = self.transition(history.state, controls, actions)
        return self.reward(history.state, controls, actions, feedbacks, next_state)

    def learn(self, history, controls, actions, feedbacks, percepts, rewards):
        # `learn` that a tick is completed and a transition has been done
        # arena is going to update the histories, it needs to know next internal state
        return self.transition(history.state, controls, actions)

    def respond(self, history, controls, actions):
        next_state = self.transition(history.state, controls, actions)
        return self.emission(history.state, controls, actions, next_state)

    # state is some feature vector extracted from previous history
    # we can not handle arbitrarily deep histories anyway
    # there seems to be a connection between the transition function and abstraction map

    # Overridable
    def transition(self, state, controls, actions):
        # choose a random but fixed function here
        return self.reset()

    # Overridable
    def emission(self, state, controls, actions, next_state):
        # pick a fixed but random function here
        return [f.random_sample() for f in self.feedback_space]

    # Overridable
    def reward(self, state, controls, actions, next_state, feedbacks):
        # choose a random but fixed function here
        return np.zeros(len(self.control_space))


class MDP(POMDP):

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.feedback_space = self.state_space

    def emission(self, state, controls, actions, next_state):
        return next_state


"""
sampling from continuous distributions:
x0', dx1', x2' ~ f(x0, dx1, x2 | h)
"""
