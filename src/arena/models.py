from collections import defaultdict

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

        # to make sure the sampled state is consistent
        # the flag is reset in `learn()` but if arena wants
        # to use this actor as a simulator, it needs to reset this flag
        # manually from outside
        self.already_sampled = False
        self.next_state = []

        # TODO: only discrete state, control and action spaces, so far
        for d in self.state_space + self.control_space + self.action_space:
            if isinstance(d, helpers.Continuous):
                raise RuntimeError(
                    "Only discrete state, control and action spaces are supported.")

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
        if not self.already_sampled:
            self.next_state = self.transition(history.state, controls, actions)
            self.already_sampled = True
        return self.reward(history.state, controls, actions, feedbacks, self.next_state)

    def learn(self, history, controls, actions, feedbacks, percepts, rewards):
        # `learn` that a tick is completed and a transition has been done
        # arena is going to update the histories, it needs to know next internal state
        if self.already_sampled:
            self.already_sampled = False
            return self.next_state
        else:
            return self.transition(history.state, controls, actions)

    def respond(self, history, controls, actions):
        if not self.already_sampled:
            self.next_state = self.transition(history.state, controls, actions)
            self.already_sampled = True
        return self.emission(history.state, controls, actions, self.next_state)

    # state is some feature vector extracted from previous history
    # we can not handle arbitrarily deep histories anyway
    # there seems to be a connection between the transition function and abstraction map

    # Important! transition, emission, and reward are the spots to
    # add general functions
    # usually the prediction is deterministic (or sampled)
    # D(Y) = f(X), and y ~ P(Y)
    # Y = [Y0, Y1, ..., Yn]
    # D(Y) = [D(Y0), D(Y1), ..., D(Yn-1)]
    # D(Y0), D(Y1|Y0), D(Y2|Y0, Y1), ..., D(Yn-1|Y0, Y1, ..., Yn-2)
    #
    # x -> N0 -> D(Y0) ~ y0
    # x,y0 -> N1 -> D(Y1) ~ y1
    # x,y0,y1 -> N3 -> D(Y2) ~ y2
    # ...
    # x,y0,y1,...,yn-2 -> Nn -> D(Yn-1) ~ yn-1
    #
    # 'bounded', 'left-bounded', 'right-bounded', 'unbounded'
    #
    # BOUNDED, LEFT_BOUNDED, RIGHT_BOUNDED, UNBOUNDED
    #
    # Discreet(0, 2, name='axis', step=1)
    # Sequence([0,1,2], name='axis')
    # g(f(X), W) = y(W)
    # E[g o f | X] = y(X) [expected outcome]
    #

    # Overridable

    def transition(self, state, controls, actions):
        # assuming state, controls and actions are indices
        condition = tuple(state + controls + actions)
        # joint distribution
        if not len(self.transition_matrix[condition]):
            dist = self.rng.random(self.num_states) + self.min_probability
            self.transition_matrix[condition] = dist / dist.sum()
        next_state = self.rng.choice(range(self.num_states),
                                     p=self.transition_matrix[condition])
        # this is a "random" sample
        # it may be different everytime the function is called
        # it is being called too many times already
        return next_state

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
        # this is a "random" sample
        # it may be different everytime the function is called
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
