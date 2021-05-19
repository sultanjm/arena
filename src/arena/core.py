import numpy as np
import abc
import helpers
from collections import defaultdict


class Arena:
    def __init__(self, *args, **kwargs):
        # local copy of args and kwargs
        self.args = args
        self.kwargs = kwargs
        # history managers for every actor
        # self.history_mgr = helpers.HistoryManager()
        self.actors = list()
        self.pre_decision_schema = list()
        self.post_decision_schema = list()
        self.order = list()
        # self.schemata[actor_A]['controls'][actor_B] = [B's control_channels idx]
        # self.schemata[actor_A]['acts'][actor_B] = [A's action_channels idx]

        # self.control_schema[actor_A][actor_B] = [A's control_channels idx]
        # self.action_schema[actor_A][actor_B] = [A's action_channels idx]
        self.schemata = defaultdict(dict)

        # self.control_edges[actor_A][actor_B] = [(A's action_channel, B's control_channel)]
        self.control_edges = defaultdict(dict)

        # for each idx in range(control_space)
        # actor.control_edges[idx] = [actor, action_channel]
        # actor.control_edges[idx] = []  # default empty, not controlled

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

            # history update for any actor is:
            # controls, actions, feedbacks, percepts
            # controls <= from preceding actors
            # actions => to succeeding actors
            # act cycle
            self.act_cycle()
            # histroy + controls --> actions (-> controls)
            # response cycle
            self.response_cycle()
            # history + controls + actions --> feedback (-> percepts)
            # evaluate each control at the history (dispatch to every controller)
            # learn cycle
            self.learn_cycle()
            # history + controls + actions + feedback + percept --> history
            # a history update is [controls, actions, feedback, percept]

    def act_cycle(self):
        # start a new decision vector
        # decision_vector[actor] = [actions for complete action_space]
        decision_matrix = defaultdict(dict)
        # for each i in actor.control_edges[idx]['action_channels']:
        # decision_vector[actor.control_edges[idx]['actor']][i]
        # go through each actor
        for actor in self.actors:
            # generate the control vector for this actor
            # provide it with the current history and control vector
            # let it react to the decision_vector
            # go over complete control space
            # get actions (controls) from another actors
            # produce random samples where necessary (for uncontrolled channels)
            control_vector = actor.random_control_vector()
            for link in actor.control_links:
                for ca_pair in link['ca_pairs']:
                    control_vector[ca_pair['control']
                                   ] = decision_matrix[link['actor']][ca_pair['action']]

            # pass the pre-decision information vector along with the history to the actor
            # the actor (re-)acts to the provided information
            action_vector = actor.act(
                self.history_mgr[actor].history, control_vector)
            # add new action information to decision information vector of this step
            decision_matrix[actor] = action_vector
        return decision_matrix

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

    def register2(self, actors):
        """
        register actors to the helpers
        initiate history managers for each actor
        """
        for actor in self.actors:
            if not isinstance(actor, Actor):
                raise RuntimeError("{} is not an actor.".format(actor))
            self.actors.append(actor)
            self.history_mgr[id(actor)] = helpers.HistoryManager(
                history_maxlen=actor.history_maxlen)

    def register(self, *actors):
        """
        register actors to the helpers
        initiate history managers for each actor
        """
        for actor in actors:
            # already registered, skip it
            if actor in self.actors:
                continue
            # register every controller of the actor
            # because the arena can't function without the controllers
            self.register(*actor.controlers.values())
            # check if the actor can be registered
            self.actors.append(actor)
            # self.history_mgr[id(actor)] = helpers.HistoryManager(history_maxlen=actor.history_maxlen)

    def deregister(self, actors): pass

    def get_schemata(self, pre_schema, post_schema=None, post_schema_is_identity=True):
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
    def __init__(self, name=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.name = 'actor_{}'.format(id(self)) if name is None else name
        self.messages = list()
        self.history_maxlen = self.kwargs.get('history_maxlen', 10)
        # actors is controlling any part of `control_space`
        # key: control_channel index
        # value: controlling actor
        self.controlers = dict()
        # actors being controlled by this actor
        # key: action_channel index
        # value: controlled actor
        self.controlled = dict()
        # controllers and controlled should be disjoint!!!

        # actors providing feedback into `percept_space`
        self.influencers = dict()

        # inward signals
        self.control_space = [helpers.Sequence()]  # a_in default control
        # [('real', math.inf), ('real', 1.0), ('natural', math.inf), ('natural', 10)]
        # [Reals(), Interval(1.0), Naturals(), Sequence(10)]
        self.percept_space = []  # e_in

        # outward signals
        self.feedback_space = [helpers.Interval()]  # a_out
        # [('natural', math.inf)]
        # [Naturals()]
        self.action_space = []  # <from second person> e_out

    def says(self, message):
        return True if self.messages.contains(message) else False

    def setup(self):
        # set percept_space and action_space for the actors being controlled from it
        # environment -> agent
        # agent.action_space = env.control_space
        # agent.percept_space = env.feedback_space
        # evn receives (control_space, percept_space) at input, (a, None)
        # env produces (feedback_space, action_space) at output, (e, None)
        return self

    def controlling(self, actor):
        """
        check through the control chain
        """
        # the actor is in the controlled list
        if actor in self.controlled.values():
            return True
        # check further down the line
        for controlled_actor in self.controlled.values():
            if controlled_actor.controlling(actor):
                return True
        # not controlled anywhere in the hierarchy
        return False

    def controlled_by(self, actor, control_channels=[]):
        # the input actor (actor) must not be controlled by the current actor (self)
        # otherwise it will create a cycle in the control mechanism
        if self.controlling(actor):
            raise RuntimeError(
                "The actor is already being controlled by the current actor.")
        # request all control channels
        if not control_channels:
            control_channels = range(len(self.control_space))
        # only valid channels are requested
        if not set(control_channels) <= set(range(len(self.control_space))):
            raise RuntimeError("The requested channels are not valid.")
        # all channels are available
        if any(key in self.controlers for key in control_channels):
            raise RuntimeError("The requested channel is already allocated.")
        # allot the requested channels
        for channel in control_channels:
            self.controlers[channel] = actor
            actor.controlled[len(actor.action_space)] = self
            actor.action_space.append(self.control_space[channel])
        # return `self` so the function calls may be chained
        return self

    def influenced_by(self, actor, feedback_channels=[], action_channels=[]):
        """
        the information stored in the history
        this is extra from it's own controls, actions, and feedbacks
        in this function the actor can ask for actions and feedbacks channels from other agents
        make sure the channels in its own control are not listed in the influenced list
        this information is populated after every agent has taken their actions
        """
        # assert actor
        if not isinstance(actor, Actor):
            raise RuntimeError("{} is not an actor.".format(actor))
        # request all channels
        if not action_channels and not feedback_channels:
            action_channels = range(len(actor.action_space))
            feedback_channels = range(len(actor.feedback_space))

        # only valid channels are requested
        if not set(action_channels) <= set(range(len(actor.action_space))):
            raise RuntimeError("The requested action channels are not valid.")
        if not set(feedback_channels) <= set(range(len(actor.feedback_space))):
            raise RuntimeError(
                "The requested feedback channels are not valid.")
        # remove any action channel already allocated to a control channel
        channels = [k for k, v in self.controlers.items() if v == actor]
        action_channels = [x for x in action_channels if x not in channels]
        for channel in action_channels:
            self.influencers[len(self.influencers)] = [
                actor, 'action', channel]
            self.percept_space.append(actor.action_space[channel])
        for channel in feedback_channels:
            self.influencers[len(self.influencers)] = [
                actor, 'feedback', channel]
            self.percept_space.append(actor.feedback_space[channel])

        return self

    def random_sample(channel):
        rng = np.random.default_rng()
        if isinstance(channel, helpers.Reals):
            return rng.standard_gamma(1)
        if isinstance(channel, helpers.Naturals):
            return rng.geometric(0.5) - 1
        if isinstance(channel, helpers.Sequence):
            return rng.choice(range(channel.len))
        if isinstance(channel, helpers.Interval):
            return rng.uniform(0.0, channel.len)

    def random_control_vector(self):
        c_vec = []
        for idx in range(len(self.control_space)):
            c_vec.append(self.random_sample(self.control_space[idx]))
        return c_vec

    def act(self, history, controls):
        # pre_decision_info should "agree" with the input space
        # output space is "received" from another actors
        pass

    def state(self, history, pre_decision_info):
        return history[-1]

    def response(self, history, pre_decision_info):
        pass

    def evaluate(self, history, pre_decision_info, decision): pass
    # the size of returned evaluation should match pre_decision_info
    # the return vector should be of the size of the control channels
    # takes in complete history -> R^[size of controls]

    def learn(self, history): pass

    def render(self):
        pass

# there might be actors which have vector controls, like factors of action space
# they can connect to multiple actors, where each actor controls a part of controls
# we should allow for a continuos control inputs with some sampling function
# important thing is to allow function approximation in the actors
# so they can keep approximate statistics
# Actor [a_out, e_out], [a_in, e_in]
# why do we need a_in and e_in?
# formally a_in and e_in can have same effect
# the actor may use e_in as as control input
# (a_in_1, a_in_2, a_in_3, ... , a_in_{K-1})
# action cycle
# [a_out, e_out] = actor.act(history, [a_in, e_in])
# evaluation cycle
# r = actor.evaluate([histroy [a_in, e_in] [a_out, e_out]])
# Dims, Continuous/discrete, bounded/unbounded
# (1, 'natural', 10)
# (1, 'real', 1.0)
# (1, 'natural', math.inf)
# (1, 'real', math.inf)
# [('real', 1.0), ('real', 1.0)] => [0.0, 1.0] x [0.0, 1.0]
# [('natural', math.inf), ('real', 2.0)] => N x [0.0, 2.0]
# [('natural', 4), ('natural', 2)] => {0, 1, 2, 3} x {0, 1}
# [('real', math.inf), ('real', math.inf)] => R x R
# it could be empty []

# design choices: spaces start from zero and always closed intervals
# the controlling agent may control a part of the control space. The rest of the controls are chosen at random.
# Important: the fixed controls are like an actor acting on it by "choosing" a fixed control.
# for example, if the control space is R x R, but the controlling actor is only choosing the first coordinate then the second
# coordinate is chosen at random. It is same to say that the agent has chosen any point on the line.

# let an helpers have 5 actors, e.g. [a0, a1, a2, a3, a4]
# from a2's perspective
# a2 is controlled by: [a0, a4] <= from pre_schema
# a2 is influenced by: [a2, a3] <= from post_schema - pre_schema
# a2 is independent of: [a1] <= not connected in any schemata

# maybe we can use post_schema directly to get the 'influence' diagram
# control dimensions are control channels. (one to many)
# in an RL helpers: agent is influenced by the environment, but not controlled by it
# the environment is controlled by the environment.
# we can use flags
# [1, 1, 0, 0] means a0 controls channel 0 and 1 of a2
# [0, 0, 1, 0] means a1 controls channel 2 of a2
# channel 3 is chosen randomly
# a2 can decide to receive a subset of (action + feedback) channels from a2 and a3, e.g. [0,0,1] and [1,0,1,0], as percepts

# history update for any actor is:
# controls, actions, feedbacks, percepts
# controls <= from preceding actors
# actions => to succeeding actors
# act cycle
# histroy + controls --> actions (-> controls)
# response cycle
# history + controls + actions --> feedback (-> percepts)
# evaluate each control at the history (dispatch to every controller)
# learn cycle
# history + controls + actions + feedback + percept --> history

# a history update is [controls, actions, feedback, percept]


if __name__ == "__main__":

    agent = Actor('Agent007')
    domain = Actor('Starship')
    domain.control_space = [helpers.Reals(), helpers.Naturals(
    ), helpers.Sequence(10), helpers.Interval(2.0)]
    domain.feedback_space = [
        helpers.Naturals(), helpers.Sequence(2), helpers.Interval(2.0)]

    domain.controlled_by(agent)
    agent.influenced_by(domain, feedback_channels=[0, 2])

    agentA = Actor('AgentA')
    agentB = Actor('AgentB')
    ab = dict()
    ab[agentA] = agentB

    pd = Actor('PD')
    pd.control_space = [helpers.Naturals(), helpers.Interval(2.0)]
    pd.feedback_space = pd.control_space

    pd.controlled_by(agentA, control_channels=[0])
    pd.controlled_by(agentB, control_channels=[1])

    agentA.influenced_by(pd, feedback_channels=[1])
    agentB.influenced_by(pd, feedback_channels=[0])

    rl = Arena()

    rl.register(agentB, pd)
    rl.register(pd)
    rl.register(agentA)

    rl.actors = ['act0', 'act1', 'act2', 'act3', 'act4']

    rl.schemata(
        pre_schema=[[0, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0]],
        post_schema=[[1, 0, 1, 0, 1],
                     [1, 1, 1, 0, 0],
                     [0, 0, 1, 0, 1],
                     [1, 1, 0, 1, 1],
                     [0, 0, 0, 0, 1]]
    )

    rl.schemata([[0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0]])

    rl.tick()

    # rl.schemata([[1, 1, 1, 1, 1],
    #              [1, 0, 1, 0, 0],
    #              [1, 0, 1, 0, 0],
    #              [1, 1, 0, 1, 1],
    #              [0, 0, 0, 0, 0]])
