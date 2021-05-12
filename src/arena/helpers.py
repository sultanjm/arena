
import collections
import itertools
import numpy as np
import copy
import enum
import abc
import math

import core


class Index(enum.Enum):
    NEXT = enum.auto()
    CURRENT = enum.auto()
    RECENT = enum.auto()
    PREVIOUS = enum.auto()
    OLDEST = enum.auto()
    FIRST = enum.auto()
    ABSOLUTE = enum.auto()


class HistoryManager:

    def __init__(self, *args, **kwargs):
        self.steplen = kwargs.get('steplen', 2)
        self.maxlen = None if not kwargs.get(
            'history_maxlen', None) else self.steplen * kwargs.get('history_maxlen', None)
        self.history = History(
            history=history, maxlen=self.maxlen, steplen=self.steplen)
        self.history.steps = kwargs.get('steps', 0.0)
        self.state_func = state_func
        self.listeners = collections.defaultdict(set)

    # def record(self, items, notify=True):
    #     steps = len(items) / self.steplen
    #     if steps and notify:
    #         self.dispatch(grl.EventType.ADD, {
    #                       'history': self.history, 'update': items})
    #     for item in items:
    #         self.history.append(item)
    #     self.history.steps += steps

    #     return self

    # def extend(self, items, notify=True):
    #     steps = len(items) / self.steplen
    #     if steps and notify:
    #         self.dispatch(grl.EventType.ADD, {
    #                       'history': self.history, 'update': items})
    #     for item in items:
    #         self.history.extension.append(item)
    #     self.history.xsteps += steps
    #     return self

    # def drop(self, steps=1.0, notify=True):
    #     if not (float(steps) * self.steplen).is_integer():
    #         raise RuntimeError(
    #             "unable to drop {} elements.".format(steps * self.steplen))
    #     dropped = list()
    #     for _ in range(int(steps*self.steplen)):
    #         dropped.append(self.history.pop())
    #     self.history.steps -= steps
    #     if dropped and notify:
    #         self.dispatch(grl.EventType.REMOVE, {
    #                       'history': self.history, 'update': dropped[::-1]})
    #     return dropped[::-1]

    # def xdrop(self, steps=None, notify=True):
    #     if steps is None:
    #         steps = self.history.xsteps
    #     dropped = list()
    #     if not (float(steps) * self.steplen).is_integer():
    #         raise RuntimeError(
    #             "unable to drop {} elements.".format(steps * self.steplen))
    #     for _ in range(int(steps*self.steplen)):
    #         dropped.append(self.history.extension.pop())
    #     self.history.xsteps -= steps
    #     if dropped and notify:
    #         self.dispatch(grl.EventType.REMOVE, {
    #                       'history': self.history, 'update': dropped[::-1]})
    #     return dropped[::-1]

    # def xmerge(self):
    #     if self.history.extension:
    #         self.record(self.xdrop(notify=False), notify=False)
    #     else:
    #         raise ValueError("The extension is empty.")

    # # @property
    # # def history(self):
    # #     return self.history

    # # @history.setter
    # # def history(self, h):
    # #     if not isinstance(h, History):
    # #         raise RuntimeError("No valid History object is provided.")
    # #     self.history = h

    # def state(self, history, *args, **kwargs):
    #     index = kwargs.get('index', Index.CURRENT)
    #     extension = kwargs.get('extension', list())
    #     history_mgr = self.assert_history_mgr(history)

    #     change = history_mgr.amend(history, index, extension)
    #     s = self.state_func(history, *args, **kwargs)
    #     history_mgr.mend(change)
    #     return s

    # def assert_history_mgr(self, history):
    #     history_mgr = self
    #     if history is not self.history:
    #         history_mgr = copy.deepcopy(self)
    #         history_mgr.history = history
    #     return history_mgr

    # def register(self, obj, event_type=EventType.ALL):
    #     self.listeners[event_type].add(obj)

    # def deregister(self, obj, event_type=grl.EventType.ALL):
    #     self.listeners[event_type].discard(obj)

    # def dispatch(self, event_type, data):
    #     evt = grl.Event(event_type, data)
    #     for obj in self.listeners[event_type]:
    #         obj.on(evt)

    # def amend(self, history, index=Index.CURRENT, extension=list()):
    #     old_h = list()
    #     old_xtn = list()
    #     # prepare history for non-current indexes
    #     if index == Index.NEXT:
    #         old_xtn = self.xdrop()
    #         self.extend(old_xtn).extend(extension)
    #     elif index == Index.PREVIOUS:
    #         old_xtn = self.xdrop(1.0)
    #         if not old_xtn:
    #             old_h = self.drop(1.0)
    #     return [old_h, old_xtn]

    # def mend(self, change):
    #     old_h, old_xtn = change
    #     # undo the changes in the history
    #     self.xdrop()
    #     self.record(old_h).extend(old_xtn)
    #     return self


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
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Interval(Space):
    def __init__(self, len=1.0):
        self.len = len
        super().__init__(self)


class Naturals(Space):
    pass


class Sequence(Space):
    def __init__(self, len=1):
        self.len = len
        super().__init__(self)


class Reals(Space):
    pass
