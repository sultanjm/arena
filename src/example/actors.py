import arena
import numpy as np


class Agent(arena.Actor):

    def setup(self, *args, **kwargs):
        return super().setup()

    def act(self, history, controls):
        return super().act(history, controls)

    def respond(self, history, controls, actions):
        return super().respond(history, controls, actions)

    def evaluate(self, history, controls, actions, feedbacks):
        return super().evaluate(history, controls, actions, feedbacks)

    def learn(self, history, evaluations):
        return super().learn(history, evaluations)

    def state(self, history, controls):
        return super().state(history, controls)

    def render(self):
        return super().render()
