import arena


class BlindMaze(arena.POMDP):

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.maze_len = kwargs.get('maze_len', 4)
        self.state_space = [arena.Sequence(
            [(x, y) for x in range(self.maze_len) for y in range(self.maze_len)])]
        self.goal_state = [self.state_space[0].labels.index(
            kwargs.get('goal_state', (0, 0)))]
        self.control_space = [arena.Sequence(['u', 'd', 'l', 'r'])]
        self.feedback_space = [arena.Sequence(range(1))]

    def transition(self, state, controls, actions):

        if state == self.goal_state:
            return self.reset()

        # inputs are indices
        a = self.control_space[0].labels[controls[0]]
        x, y = self.state_space[0].labels[state[0]]

        if a == 'u':
            s_next = (x, min(y + 1, self.maze_len - 1))
        elif a == 'd':
            s_next = (x, max(y - 1, 0))
        elif a == 'l':
            s_next = (max(x - 1, 0), y)
        elif a == 'r':
            s_next = (min(x + 1, self.maze_len - 1), y)
        else:
            s_next = (x, y)

        return [self.state_space[0].labels.index(s_next)]

    def reward(self, state, controls, actions, next_state, feedbacks):
        return [1.0] if state == self.goal_state else [0.0]

    def emission(self, state, controls, actions, next_state):
        # uninformative feedback
        return [0]
