import abc
import numpy as np


class Arena:
    def __init__(self, *args, **kwargs):
        # local copy of args and kwargs
        self.args = args
        self.kwargs = kwargs
        # history managers for every actor
        # self.history_mgr = arena.HistoryManager()
        self.actors = list()
        self.pre_decision_schema = list()
        self.post_decision_schema = list()
        self.order = list()

    def tick(self, steps=1):
        """
        go through each registered actor's act function
        prepare pre-decision information vector for each actor
        prepare post-decision information updates
        extend histories with post-decision information
        """
        for step in range(steps):
            # populate the decision vector from all actors
            decision_vector = self.interact()
            # update histories using the current decision vector
            self.update(decision_vector)

    def interact(self):
        # make a (None) decision vector
        decision_vector = [None] * len(self.actors)
        # go through each actor
        for idx, actor_id in enumerate(self.order):
            # get the sorted by order mask for pre-decision information for the actor
            pre_decision_mask = (self.pre_decision_schema[actor_id])[
                self.order]
            # extract only the "active" information bits
            pre_decision_info = [decision_vector[x]
                                 for x in range(len(self.actors)) if pre_decision_mask[x]]
            # pass the pre-decision information vector along with the history to the actor
            # the actor (re-)acts to the provided information
            action = self.actors[actor_id].act(
                self.history_mgr.history(self.actors[actor_id]), pre_decision_info)
            # add new action information to decision information vector of this step
            decision_vector[idx] = action
        return decision_vector

    def update(self, decision_vector):
        # all registered actors have taken their actions for this step
        for idx, actor_id in enumerate(self.order):
            # get the sorted by order mask for post-decision information for the actor
            post_decision_mask = (self.post_decision_schema[actor_id])[
                self.order]
            # extract only the "active" information bits
            post_decision_info = [decision_info[x]
                                  for x in range(len(self.actors)) if post_decision_mask[x]]
            # pass the post-decision information vector along with the history to the actor
            # go through the decision information vector to update history for each actor
            self.history_mgr.update(self.actors[actor_id], post_decision_info)
            # invoke learn() function for each actor
            self.actors[actor_id].learn(
                self.history_mgr.history(self.actors[actor_id]))

    def register(self, actors):
        """
        register actors to the arena
        initiate history managers for each actor
        """
        for actor in self.actors:
            if not isinstance(actor, Actor):
                raise RuntimeError("{} is not an actor.".format(actor))
            self.actors.append(actor)
            self.history_mgr[id(actor)] = arena.HistoryManager(
                history_maxlen=actor.history_maxlen)

    def deregister(self, actors): pass

    def schemata(self, pre_schema, post_schema=None, post_schema_is_identity=True):
        # each actor can indicate to observe other actors' actions
        # a valid allocation should not have any self-observations (i.e. the diagonal should be zero)
        # make sure schema is of valid size
        # calculate the execution order
        # a0: 0 0 0
        # a1: 1 0 0
        # en: 0 1 0
        # {0:actor1, 1:actor4, 2:actor0}
        # {0:[], 1:[actor1], 2:[actor1,actor4]}
        # extract "default" information schema from order schema

        # a valid order schema should satisfy the following conditions:
        # 1. there must exist at least one zero column, (~schema.T.any(axis=1)).sum() > 0
        # 2. there must exist at least one zero row, (~schema.any(axis=1)).sum() > 0
        # 3. the diagonal must be zero, schema.trace() = 0
        # 4. there is no "race" between actor, which means cross diagonal values of
        #    any (non-terminal) pair of actors are not simultaneously active, np.bitwise_and(schema,schema.T).sum() == 0

        pre_schema = np.array(pre_schema)

        if pre_schema.shape != [len(self.actors)] * 2:
            raise RuntimeError(
                "Pre-decision schema does not have valid dimensions.")

        if (~pre_schema.any(axis=1)).sum() == 0:
            raise RuntimeError(
                "Order schema does not have any zero row. There is no actor to put at the start of the order.")

        if (~pre_schema.T.any(axis=1)).sum() == 0:
            raise RuntimeError(
                "Order schema does not have any zero column. There is no actor to put at the end of the order.")

        if pre_schema.trace() != 0:
            raise RuntimeError(
                "Order schema has non-zero trace. There is at least one actor which is observing its own action.")

        if np.bitwise_and(pre_schema, pre_schema.T).sum() != 0:
            raise RuntimeError(
                "There is a race between a pair of actors. They both want to observe each others actions.")

        # after validating, we can follow the blew (neat) algorithm to get an order
        # note: the order is non-unique
        # get the index (idx) of ANY zero row
        # append this index in a list
        # remove this row and corresponding column, np.delete(np.delete(schema, idx, 0), idx, 1)
        # continue until no more entries are left

        tmp_actors = list(range(len(pre_schema)))
        tmp_schema = np.copy(pre_schema)
        self.order = list()
        while len(tmp_actors):
            idx = np.where(~tmp_schema.any(axis=1))[0][0]
            self.order.append(tmp_actors[idx])
            tmp_actors = np.delete(tmp_actors, idx)
            tmp_schema = np.delete(np.delete(tmp_schema, idx, 0), idx, 1)

        # save a local copy of the schema
        self.pre_schema = pre_schema
        # afterwards, we can access the pre-decision information set of each actor as
        # (self.schema[self.order[idx]])[self.order]
        # for each idx
        # check if post-decision information schema is provided.
        if post_schema is None:
            if post_schema_is_identity:
                # we use "all observations" as default
                post_schema = np.ones([len(self.actors)] * 2)
            else:
                # this schema is always pre-decision information schema + identity + extra info
                post_schema = pre_schema + np.eye(len(self.actors))

        if post_schema.shape != [len(self.actors)] * 2:
            raise RuntimeError(
                "Post-decision schema does not have valid dimensions.")

        if post_schema.trace() != len(self.actors):
            raise RuntimeError(
                "Post-decision schema has invalid trace. Each actor should have its own action as a post-decision information.")

        if not np.all(np.multiply((pre_schema != 0), post_schema) == (pre_schema != 0)):
            raise RuntimeError(
                "Post-decision schema does not contain pre-decision schema.")

        # save a local copy of post-decision information schema
        self.post_schema = post_schema

    def stats(self): pass


class Actor(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.messages = list()
        self.history_maxlen = self.kwargs.get('history_maxlen', 10)

    def says(self, message):
        return True if self.messages.contains(message) else False

    def setup(self): pass

    def act(self, history, pre_decision_info):
        pass

    def state(self, history, pre_decision_info):
        return history[-1]

    def evaluate(self, history, pre_decision_info, decision): pass
    # the size of returned evaluation should match pre_decision_info

    def learn(self, history): pass

    def render(self):
        pass


__name__ == "main"

rl = Arena()
rl.actors = ['act0', 'act1', 'act2', 'act3', 'act4']
# rl.pre_decision_info_schema([[0, 0, 1, 0, 1],
#                              [1, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 1],
#                              [1, 1, 0, 0, 1],
#                              [0, 0, 0, 0, 0]])

rl.pre_decision_info_schema([[0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1],
                             [1, 1, 0, 0, 1],
                             [0, 0, 0, 0, 0]])

rl.tick()

# rl.post_decision_info_schema([[1, 1, 1, 1, 1],
#                               [1, 0, 1, 0, 0],
#                               [1, 0, 1, 0, 0],
#                               [1, 1, 0, 1, 1],
#                               [0, 0, 0, 0, 0]])
