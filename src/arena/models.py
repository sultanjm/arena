from collections import defaultdict

from arena.utils import random_probability_matrix

from . import core
from . import helpers

import numpy as np


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

        # TODO: only bounded spaces, so far
        for d in self.state_space + self.control_space + self.feedback_space:
            if isinstance(d, helpers.Reals) or isinstance(d, helpers.Naturals):
                raise RuntimeError("Only bounded spaces are supported.")

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

        # n1, n2, n3
        self.total_bins = 0
        for s in self.state_space:
            self.total_bins += s.bins

        # only for random (PO)MDPs
        self.transition_matrix = defaultdict(list)
        self.emission_matrix = defaultdict(list)
        self.reward_matrix = defaultdict(list)

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
        # get bin values
        control_bins = self.binify(controls, self.control_space)
        action_bins = self.binify(actions, self.action_space)
        state_bins = self.binify(state, self.state_space)
        condition = tuple(state_bins + control_bins + action_bins)
        if not self.transition_matrix[condition]:

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

    def binify(self, values, space):
        return [space[idx].bin(values[idx]) for idx in range(len(space))]


class MDP(POMDP):

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.feedback_space = self.state_space

    def emission(self, state, controls, actions, next_state):
        return next_state


"""
CDF of a mixed random variable
state can be a mixed multi-dimensional RV
conditioned on multi-dimensional mixed variables
lets start by [controls + actions] as a single set of inputs
empty actions is no more a distinct case
e.g. 
N x R x I x S == inputs
N x R x I x S == outputs
lets limit to bounded parameters to begin with
so,
I x S only
state can only be a collection of sequences and intervals
lets limit to uniform distributions on intervals
problem: how to fix a realized distribution?
assumption: smoothness
lets chop intervals into `delta` bins

"""
