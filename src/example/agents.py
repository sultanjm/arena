import arena
import numpy as np

from collections import defaultdict


class GreedyQAgent(arena.Actor):

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.g = kwargs.get('discount_factor', 0.999)
        # optimistic initialization
        self.def_val = 1/(1-self.g) if not kwargs.get('initial_q_value',
                                                      None) else kwargs.get('initial_q_value', None)
        self.Q = defaultdict(list)
        self.eps = kwargs.get('exploration_factor', 1e-3)
        # set by first act() call
        self.action_space_shape = []
        self.num_actions = None
        self.state_space_shape = []
        self.num_states = None

        self.rng = np.random.default_rng()

    def act(self, history, controls):
        state_idx = np.ravel_multi_index(history.state, self.state_space_shape)
        if not self.Q[state_idx]:
            # create this row
            self.Q[state_idx] = [self.def_val] * self.num_actions
        action_idx = argmax(self.Q[state_idx])
        # epsilon greedy action
        if self.rng.uniform() < self.eps:
            action_idx = self.rng.choice(range(self.num_actions))
        return list(np.unravel_index(action_idx, self.action_space_shape))

    # history.state.idx
    # controls.idx
    # actions.idx

    def learn(self, history, controls, actions, feedbacks, percepts, rewards):
        state = np.ravel_multi_index(history.state, self.state_space_shape)
        next_state_idx = self.state_map(history, controls, actions,
                                        feedbacks, percepts, rewards)
        next_state = np.ravel_multi_index(
            next_state_idx, self.state_space_shape)
        if not self.Q[next_state]:
            # create this row
            self.Q[next_state] = [self.def_val] * self.num_actions
        action = np.ravel_multi_index(actions, self.action_space_shape)
        self.Q[state][action] = self.Q[state][action] + 0.999 * \
            (average(rewards) + self.g *
             max(self.Q[next_state]) - self.Q[state][action])
        return next_state_idx

    # reset() is called after registration with arena
    def reset(self):
        if not self.action_space_shape:
            self.action_space_shape = [
                self.action_space[idx].len for idx in range(len(self.action_space))]
            self.num_actions = np.prod(self.action_space_shape)
        if not self.state_space_shape:
            # percept space is the state space
            # self.state_space = self.percept_space
            # last two actions make the action space
            self.state_space = self.action_space * 2
            self.state_space_shape = [
                self.state_space[idx].len for idx in range(len(self.state_space))]
            self.num_states = np.prod(self.state_space_shape)
        state = self.rng.choice(range(self.num_states))
        return list(np.unravel_index(state, self.state_space_shape))

    def state_map(self, history, controls, actions, feedbacks, percepts, rewards):
        # percepts are the states
        # most recent two actions are the state
        next_state = history.state[-1::]
        next_state = next_state + actions
        return next_state
