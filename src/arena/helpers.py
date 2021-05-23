from . import core
from . import utils

import collections
import numpy as np
import enum
import abc


class Index(enum.Enum):
    NEXT = enum.auto()
    CURRENT = enum.auto()
    RECENT = enum.auto()
    PREVIOUS = enum.auto()
    OLDEST = enum.auto()
    FIRST = enum.auto()
    ABSOLUTE = enum.auto()


class History(collections.deque):
    __slots__ = 'steps'


class HistoryManager:

    def __init__(self, *args, **kwargs):
        # save a local copy of args kwargs
        self.args = args
        self.kwargs = kwargs

        self.history_maxlen = self.kwargs.get('history_maxlen', None)
        self.listeners = collections.defaultdict(set)

        # dynamic queue (default length: None:maxlen)
        self.history = History(self.kwargs.get(
            'history', list()), self.history_maxlen)
        self.history.steps = 0  # steps of history

        self.control_vector = []
        self.action_vector = []
        self.feedback_vector = []
        self.percept_vector = []

    def record(self):
        # control, action, feedback, percept
        self.history.append(self.control_vector + self.action_vector +
                            self.feedback_vector + self.percept_vector)
        self.history.steps += 1
        self.control_vector = []
        self.action_vector = []
        self.feedback_vector = []
        self.percept_vector = []
        return self


class PerceptManager:

    def __init__(self, emission_func=lambda s: s, percept_space=None, percept=None):
        self.emission_func = emission_func
        self.percept_space = percept_space
        self.percept = percept

    def perception(self, s):
        if not callable(self.emission_func):
            raise RuntimeError("No valid percept function is provided.")
        return self.emission_func(s)


class StateManager:

    def __init__(self, *args, **kwargs):
        # keep a local copy of args and kwargs
        self.args = args
        self.kwargs = kwargs
        # state space
        self.state_space = self.kwargs.get('state_space', None)
        # starting state
        self.current_state = self.kwargs.get('start_state', None)
        # history to state map
        self.state_func = self.kwargs.get('state_func', lambda history: None)
        # record the starting state
        self.state_history_mgr.record([self.current_state])
        # state transition function (if required)
        self.transition_func = self.kwargs.get(
            'transition_func', lambda s, a: s)
        # internal record keeping of the state history
        self.state_history_mgr = HistoryManager(
            maxlen=self.kwargs.get('history_maxlen', 1))

    def get_state(self, history, index=Index.CURRENT):
        extension = kwargs.get('extension', list())
        history_mgr = self.assert_history_mgr(history)

        change = history_mgr.amend(history, index, extension)
        s = self.state_func(history)
        history_mgr.mend(change)
        return s

    def simulate(self, action, state=None):
        if state is None:
            state = self.current_state
        if not callable(self.transition_func):
            raise RuntimeError("No valid transition function is provided.")
        return self.transition_func(state, action)

    def transit(self, action):
        if action is not None:
            self.current_state = self.simulate(action)
            self.state_history_mgr.record([action, self.current_state])
        return self.current_state


class ActionManager:

    def __init__(self, action_space=None, action=None):
        self.action_space = action_space
        self.action = action


class RewardManager:

    def __init__(self, reward_func=lambda h, *x, **y: 0):
        self.reward_func = reward_func
        # use local history manager to manipulate input histories
        self.history_mgr = grl.HistoryManager()

    def reward(self, history, *args, **kwargs):
        self.history_mgr.history = history
        extension = kwargs.get('extension', list())
        index = kwargs.get('index', Index.CURRENT)
        change = self.history_mgr.amend(history, index, extension)

        r = self.reward_func(history)

        self.history_mgr.mend(change)
        return r


class Space(abc.ABC):
    def __init__(self, name=None):
        self.name = 'id:{}'.format(id(self)) if name is None else name

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
        return np.random.default_rng().choice(range(self.len))


class Interval(Space):
    def __init__(self, range=(0.0, 1.0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = min(range)
        self.max = max(range)
        self.len = self.max - self.min

    def random_sample(self):
        return np.random.default_rng().uniform(self.min, self.max)


class Naturals(Space):
    def random_sample(self):
        return np.random.default_rng().geometric(0.5) - 1


class Reals(Space):
    def random_sample(self):
        return np.random.default_rng().standard_gamma(1)
