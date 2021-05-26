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
        # state-space is internal to the actor
        # outside world should not worry about this space
        # it's the actor who knows what it is and how to manipulate it
        # it should be specified and *must* be different from ``self.state_space``

        # TODO: only bounded spaces, so far
        for d in self.state_space + self.control_space + self.feedback_space:
            if isinstance(d, helpers.Reals) or isinstance(d, helpers.Naturals):
                raise RuntimeError("Only bounded spaces are supported.")

        self.state_space_shape = [
            self.state_space[idx].len for idx in range(len(self.state_space))]
        self.num_states = np.prod(self.state_space_shape)
        self.feedback_space_shape = [
            self.feedback_space[idx].len for idx in range(len(self.feedback_space))]
        self.num_feedbacks = np.prod(self.feedback_space_shape)

        # random number generator
        self.rng = np.random.default_rng()

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
        # assuming state, controls and actions are indices
        condition = tuple(state + controls + actions)
        # joint distribution
        if not len(self.transition_matrix[condition]):
            dist = self.rng.random(self.num_states) + self.min_probability
            self.transition_matrix[condition] = dist / dist.sum()
        idx = self.rng.choice(range(self.num_states),
                              p=self.transition_matrix[condition])
        return list(np.unravel_index(idx, self.state_space_shape))

    # Overridable
    def emission(self, state, controls, actions, next_state):
        # assuming state, controls, actions and next_state are indices
        condition = tuple(state + controls + actions + next_state)
        # joint distribution
        if not len(self.emission_matrix[condition]):
            dist = self.rng.random(self.num_feedbacks)
            self.emission_matrix[condition] = dist / dist.sum()
        idx = self.rng.choice(range(self.num_feedbacks),
                              p=self.emission_matrix[condition])
        return list(np.unravel_index(idx, self.feedback_space_shape))

    # Overridable
    def reward(self, state, controls, actions, next_state, feedbacks):
        # assuming state, controls, actions, next_state and feedbacks are indices
        condition = tuple(state + controls + actions + next_state + feedbacks)
        if not self.reward_matrix[condition]:
            self.reward_matrix[condition] = self.rng.random(
                len(self.control_space))
        return self.reward_matrix[condition]


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
