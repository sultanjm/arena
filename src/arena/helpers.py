from . import core
from . import utils

import collections
import numpy as np
import abc

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


# Space(
#   Discrete([1,4,5], 'x-axis'),
#   Continuous(1.9, 2.4, 'y-axis')
#   Discrete(range(5), 'z-axis'),
#   Continuous(-np.inf, 2.4, 'k-axis'),
#   Continuous(1.9, np.inf, 'n-axis'),
#   Continuous(-np.inf, np.inf, 'm-axis'),
#   Discrete(np.arange(0.0, 1.0, 1e-3), 'o-axis')
# )
#
# Space::random_sample()
# global step resolution
# space = Space([],range(1),(0.0,1.0,1e-6),labels=['x','y','z'])

# idx = space.random_sample()['index']
# label = space.random_sample()['label']
# label = space.label(idx)
# idx = space.index(label)
# ur_idx = space.unravel(idx)
# idx = space.ravel(ur_idx)

# class Space:
#     def __init__(self, *axes, name=None):
#         self.name = f"space:{id(self)}" if not name else name
#         self.rng = np.random.default_rng()
#         self.axes = []
#         self.add(*axes, labels=labels)

#     def add(self, *axes):
#         for idx, axis in enumerate(axes):
#             if not isinstance(axis, collections.Iterable):
#                 raise RuntimeError(f"{axis} not a a valid axis.")
#             type = 'sequence'
#             if isinstance(axis, tuple):
#                 # interval
#                 type = 'interval'
#                 if len(axis) < 3:
#                     # no step size provided, use default
#                     axis = tuple(list(axis) + [self.step_size])
#             label = f"axis:{len(self.axes)}" if not labels else labels[idx]
#             self.axes.append(axis)
#             self.labels.append(label)
#             self.types.append(type)
#         self.calculate_shape()

#     def calculate_shape(self):
#         self.shape = [None] * len(self.axes)
#         for idx, axis in enumerate(self.axes):
#             if self.types[idx] == 'interval':
#                 self.shape[idx] = int(
#                     np.ceil((max(axis[0:2])-min(axis[0:2]))/axis[2]))
#             else:
#                 self.shape[idx] = len(axis)
#         self.size = int(np.prod(self.shape))

#     def remove(self, *indices):
#         raise NotImplementedError

#     # retruns the index
#     def random_sample(self):
#         return self.unravel(self.rng.integers(self.size))

#     def label(self, indices):
#         lbl = []
#         for idx, value in enumerate(indices):
#             if self.types[idx] == 'interval':
#                 lbl.append(min(self.axes[idx][0:2]) + value*self.axes[idx][2])
#             else:
#                 lbl.append(self.axes[idx][value])
#         return lbl

#     def ravel(self, ur_idx):
#         return np.ravel_multi_index(ur_idx, self.shape)

#     def unravel(self, idx):
#         return np.unravel_index(idx, self.shape)


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


# class Discrete(Axis):
#     def __init__(self, left, right, name=None, step=1):
#         super().__init__(name)
#         self.left = min(left, right)
#         self.right = max(left, right)
#         self.bound = Axis.boundedness(self.left, self.right)

#     def random_sample(self):
#         if self.type == BOUNDED:
#             return self.rng.uniform(self.left, self.right)
#         elif self.type == UNBOUNDED:
#             return self.rng.standard_normal()
#         elif self.type == LEFT_BOUNDED:
#             return self.left + self.rng.exponential()
#         elif self.type == RIGHT_BOUNDED:
#             return self.right - self.rng.exponential()


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

# class Dimension(abc.ABC):
#     def __init__(self, name=None):
#         self.name = 'dimension:{}'.format(id(self)) if name is None else name
#         self.rng = np.random.default_rng()

#     @abc.abstractmethod
#     def random_sample(self):
#         pass


# # control_space = [Reals('pos_x'), Reals('pos_y')]
# # feedback_space = [Sequence(['okay', 'danger'], name='status'), Sequence(len=10, name='levels')]
# # feedback_space = [Sequence(10,'levels')]

# # Sequence.labels(['x','y'], 'status')
# # Sequence(range(10), 'levels')
# # Sequence(['okay', 'danger'], 'status')


# class Sequence(Dimension):
#     def __init__(self, labels=[], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.labels = labels
#         self.len = len(self.labels)

#     def random_sample(self):
#         # random sample is the index not the label
#         return self.rng.choice(range(self.len))


# class Interval(Dimension):
#     def __init__(self, range=(0.0, 1.0), resolution=1e-2, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.min = min(range)
#         self.max = max(range)
#         self.width = self.max - self.min
#         self.resolution = resolution
#         self.labels = np.arange(self.min, self.max, self.resolution)
#         self.len = len(self.labels)

#     def random_sample(self):
#         return self.rng.choice(range(self.len))
#         # return self.rng.uniform(self.min, self.max)


# class Naturals(Dimension):
#     def __init__(self, name):
#         raise NotImplementedError

#     def random_sample(self):
#         return self.rng.geometric(0.5) - 1


# class Reals(Dimension):
#     def __init__(self, name):
#         raise NotImplementedError

#     def random_sample(self):
#         return self.rng.standard_gamma(1)
