from . import core
from . import helpers

import numpy as np


class MDP(core.Actor):

    def setup(self, *args, **kwargs):
        self.num_actions = kwargs.get('num_actions', 2)
        self.num_states = kwargs.get('num_states', 2)
        self.min_probability = kwargs.get('min_probability', 0.0)
        self.control_space = [helpers.Sequence(range(self.num_actions))]
        self.feedback_space = [helpers.Sequence(range(self.num_states))]
        # random number generator
        self.rng = np.random.default_rng()
        self.transition_matrix = kwargs.get('transition_matrix', None)
        if not self.transition_matrix:
            self.transition_matrix = self.rng.random([
                self.num_states, self.num_actions, self.num_states]) + self.min_probability  # [s][a][s_nxt]
            # normalize to get a stochastic matrix
            self.transition_matrix = self.transition_matrix / \
                self.transition_matrix.sum(axis=-1, keepdims=True)
        self.reward_matrix = kwargs.get('reward_matrix', None)
        if not self.reward_matrix:
            self.reward_matrix = self.rng.random(
                [self.num_states, self.num_actions, self.num_states])  # [s][a][s_nxt]

    def respond(self, history, controls, actions):
        state = self.state(history, controls)
        return [self.rng.choice(self.feedback_space[0].range, p=self.transition_matrix[state[0]][controls[0]])]

    def state(self, history, controls):
        if not len(history):
            # provide a random state if initiated with empty history
            return [self.feedback_space[0].random_sample()]

        last_step = history[-1]
        # last_step = controls + actions + feedbacks + percepts
        offset = len(self.control_space) + len(self.action_space)
        return last_step[offset:offset + len(self.feedback_space)]

    def evaluate(self, history, controls, actions, feedbacks):
        state = self.state(history, controls)
        return [self.reward_matrix[state[0]][controls[0]][feedbacks[0]]]
