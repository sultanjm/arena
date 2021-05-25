from . import core
from . import utils

import collections
import numpy as np
import abc


class History(collections.deque):
    """
    A dynamic queue is used as a history object
    steps: actual number of steps
    state: current internal state corresponds to the complete history
    """
    __slots__ = 'steps', 'state'


class HistoryManager:

    def __init__(self, *args, **kwargs):
        self.history_maxlen = self.kwargs.get('history_maxlen', None)
        self.listeners = collections.defaultdict(set)

        # dynamic queue (default length: None:maxlen)
        self.history = History(self.kwargs.get(
            'history', list()), self.history_maxlen)
        if not self.history.steps:
            self.history.steps = 0  # steps of history

    def record(self, controls, actions, feedbacks, percepts):
        # control, action, feedback, percept
        self.history.append(controls + actions + feedbacks + percepts)
        self.history.steps += 1
        return self


class Space(abc.ABC):
    def __init__(self, name=None):
        self.name = 'id:{}'.format(id(self)) if name is None else name
        self.rng = np.random.default_rng()

    @abc.abstractmethod
    def random_sample(self):
        pass


# control_space = [Reals('pos_x'), Reals('pos_y')]
# feedback_space = [Sequence(['okay', 'danger'], name='status'), Sequence(len=10, name='levels')]
# feedback_space = [Sequence(10,'levels')]

# Sequence.labels(['x','y'], 'status')
# Sequence(range(10), 'levels')
# Sequence(['okay', 'danger'], 'status')


class Sequence(Space):
    def __init__(self, labels=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.len = len(self.labels)

    def random_sample(self):
        # random sample is the index not the label
        return self.rng.choice(range(self.len))


class Interval(Space):
    def __init__(self, range=(0.0, 1.0), resolution=1e-2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = min(range)
        self.max = max(range)
        self.width = self.max - self.min
        self.resolution = resolution
        self.labels = np.arange(self.min, self.max, self.resolution)
        self.len = len(self.labels)

    def random_sample(self):
        return self.rng.uniform(self.min, self.max)


class Naturals(Space):
    def random_sample(self):
        return self.rng.geometric(0.5) - 1


class Reals(Space):
    def random_sample(self):
        return self.rng.standard_gamma(1)
