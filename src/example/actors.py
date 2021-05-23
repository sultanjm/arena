import arena


class Agent(arena.Actor):

    def setup(self):
        return super().setup()

    def act(self, history, controls):
        return super().act(history, controls)

    def respond(self, history, controls, actions):
        return super().respond(history, controls, actions)

    def evaluate(self, history, controls, actions, feedbacks):
        return super().evaluate(history, controls, actions, feedbacks)

    def learn(self, history):
        return super().learn(history)

    def state(self, history, controls):
        return super().state(history, controls)

    def render(self):
        return super().render()
