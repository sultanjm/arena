import collections
import numpy as np

BOUNDED = 0
LEFT_BOUNDED = 1
RIGHT_BOUNDED = 2
UNBOUNDED = 4


class History(collections.deque):
    """
    A dynamic queue is used as a history object
    steps: actual number of steps
    state: current internal state corresponds to the complete history
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.state = []


class HistoryManager:

    def __init__(self, *args, **kwargs):
        self.history_maxlen = kwargs.get('history_maxlen', 10)
        # dynamic queue (default length: None:maxlen)
        self.history = History(kwargs.get(
            'history', list()), self.history_maxlen)

    def record(self, controls, actions, feedbacks, percepts):
        # control, action, feedback, percept
        self.history.append(controls + actions + feedbacks + percepts)
        self.history.steps += 1
        return self


class Axis:

    def __init__(self, name=None):
        self.name = f"axis:{id(self)}" if not name else name
        self.rng = np.random.default_rng()

    def sample(self):
        raise NotImplementedError

    @staticmethod
    def boundedness(left, right):
        type = None
        if np.isneginf(left) and np.isposinf(right):
            type = UNBOUNDED
        elif np.isneginf(left) and not np.isposinf(right):
            type = RIGHT_BOUNDED
        elif not np.isneginf(left) and np.isposinf(right):
            type = LEFT_BOUNDED
        elif not np.isneginf(left) and not np.isposinf(right):
            type = BOUNDED
        return type


class Category(Axis):
    def __init__(self, values, name=None):
        super().__init__(name)
        self.values = values

    def sample(self):
        return self.rng.choice(len(self.values))


class Continuous(Axis):
    def __init__(self, left, right, name=None):
        super().__init__(name)
        self.left = min(left, right)
        self.right = max(left, right)
        self.type = Axis.boundedness(self.left, self.right)

    def sample(self):
        if self.type == BOUNDED:
            return self.rng.uniform(self.left, self.right)
        elif self.type == UNBOUNDED:
            return self.rng.standard_normal()
        elif self.type == LEFT_BOUNDED:
            return self.left + self.rng.exponential()
        elif self.type == RIGHT_BOUNDED:
            return self.right - self.rng.exponential()


class Discrete(Continuous):
    def __init__(self, left, right, name=None, step=1, var=1000):
        super().__init__(left, right, name=name)
        self.var = var
        self.step = step
        if self.type == BOUNDED:
            self.len = int(np.ceil((self.right-self.left)/self.step))

    def random_integer(self):
        # Skellam distribution: zero mode high variance distribution
        return self.rng.poisson(self.var) - self.rng.poisson(self.var)

    def sample(self):
        if self.type == BOUNDED:
            return self.left + self.rng.integers(self.len)*self.step
        elif self.type == UNBOUNDED:
            return self.random_integer()
        elif self.type == LEFT_BOUNDED:
            return self.left + abs(self.random_integer())*self.step
        elif self.type == RIGHT_BOUNDED:
            return self.right - abs(self.random_integer())*self.step
