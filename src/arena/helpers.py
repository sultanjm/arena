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


# Space(
#   Sequence(),
#   Interval()
# )
#
# global step resolution
# space = Space([],range(1),(0.0,1.0,1e-6),labels=['x','y','z'])

# idx = space.random_sample()['index']
# label = space.random_sample()['label']
# label = space.label(idx)
# idx = space.index(label)
# ur_idx = space.unravel(idx)
# idx = space.ravel(ur_idx)

class Space:
    def __init__(self, *axes, labels=None, name=None, step_size=1e-6):
        self.name = f"space:{id(self)}" if not name else name
        self.step_size = step_size
        self.rng = np.random.default_rng()
        self.axes = []
        self.types = []
        self.labels = []
        self.shape = []
        self.size = 0
        self.add(*axes, labels=labels)

    def add(self, *axes, labels=None):
        for idx, axis in enumerate(axes):
            if not isinstance(axis, collections.Iterable):
                raise RuntimeError(f"{axis} not a a valid axis.")
            type = 'sequence'
            if isinstance(axis, tuple):
                # interval
                type = 'interval'
                if len(axis) < 3:
                    # no step size provided, use default
                    axis = tuple(list(axis) + [self.step_size])
            label = f"axis:{len(self.axes)}" if not labels else labels[idx]
            self.axes.append(axis)
            self.labels.append(label)
            self.types.append(type)
        self.calculate_shape()

    def calculate_shape(self):
        self.shape = [None] * len(self.axes)
        for idx, axis in enumerate(self.axes):
            if self.types[idx] == 'interval':
                self.shape[idx] = int(
                    np.ceil((max(axis[0:2])-min(axis[0:2]))/axis[2]))
            else:
                self.shape[idx] = len(axis)
        self.size = int(np.prod(self.shape))

    def remove(self, *indices):
        raise NotImplementedError

    # retruns the index
    def random_sample(self):
        return self.unravel(self.rng.choice(range(self.size)))

    def label(self, indices):
        lbl = []
        for idx, value in enumerate(indices):
            if self.types[idx] == 'interval':
                lbl.append(min(self.axes[idx][0:2]) + value*self.axes[idx][2])
            else:
                lbl.append(self.axes[idx][value])
        return lbl

    def ravel(self, ur_idx):
        return np.ravel_multi_index(ur_idx, self.shape)

    def unravel(self, idx):
        return np.unravel_index(idx, self.shape)


class Dimension(abc.ABC):
    def __init__(self, name=None):
        self.name = 'dimension:{}'.format(id(self)) if name is None else name
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


class Sequence(Dimension):
    def __init__(self, labels=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.len = len(self.labels)

    def random_sample(self):
        # random sample is the index not the label
        return self.rng.choice(range(self.len))


class Interval(Dimension):
    def __init__(self, range=(0.0, 1.0), resolution=1e-2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = min(range)
        self.max = max(range)
        self.width = self.max - self.min
        self.resolution = resolution
        self.labels = np.arange(self.min, self.max, self.resolution)
        self.len = len(self.labels)

    def random_sample(self):
        return self.rng.choice(range(self.len))
        # return self.rng.uniform(self.min, self.max)


class Naturals(Dimension):
    def __init__(self, name):
        raise NotImplementedError

    def random_sample(self):
        return self.rng.geometric(0.5) - 1


class Reals(Dimension):
    def __init__(self, name):
        raise NotImplementedError

    def random_sample(self):
        return self.rng.standard_gamma(1)
