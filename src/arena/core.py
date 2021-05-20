import numpy as np
import abc
import helpers
from collections import defaultdict
import utils


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

        self.history_mgr = defaultdict(helpers.HistoryManager)

        # structure:
        #   decision_matrix[actor] = [action vector from action_space]
        #   response_matrix[actor] = [feedback vector from feedback_space]
        #   percept_matrix[actor]  = [action & feedback vector from percept_space]
        self.decision_matrix = defaultdict(list)
        self.response_matrix = defaultdict(list)
        self.percept_matrix = defaultdict(list)
        self.steps = 0

    def tick(self, steps=1):
        """
        go through each registered actor's act function
        prepare pre-decision information vector for each actor
        prepare post-decision information updates
        extend histories with post-decision information
        """
        for step in range(steps):
            # populate the decision vector from all actors

            # history update for any actor is:
            # controls, actions, feedbacks, percepts
            # controls <= from preceding actors
            # actions => to succeeding actors
            # act cycle
            self.act_cycle()
            # histroy + controls --> actions (-> controls)
            # history + controls + actions --> feedback (-> percepts)
            # evaluate each control at the history (dispatch to every controller)
            # learn cycle
            self.learn_cycle()
            # history + controls + actions + feedback + percept --> history
            # a history update is [controls, actions, feedback, percept]
            # a history tuple is of len(control_space) + len(action_space) + len(feedback_space) + len(percept_space) long
            self.steps += 1

    def act_cycle(self):
        # start a new decision matrix
        # structure:
        #   decision_matrix[actor] = [action vector from action_space]
        #   response_matrix[actor] = [feedback vector from feedback_space]
        #   percept_matrix[actor]  = [action & feedback vector from percept_space]
        decision_matrix = defaultdict(list)
        response_matrix = defaultdict(list)
        percept_matrix = defaultdict(list)
        # go through each actor
        # assuming the actors are in order
        for actor in self.actors:
            # generate a "random" control vector just in case nobody is controling
            control_vector = [utils.random_sample(
                c) for c in actor.control_space]
            # check the links
            # if the actor is being controlled
            for controller in actor.inward_control_links.keys():
                # there is a controller
                # the links are stored as control-action channel pairs
                for c_ch, a_ch in actor.inward_control_links[controller]:
                    # get the action from controller to the actor
                    control_vector[c_ch] = decision_matrix[controller][a_ch]
            # update control_vector for current cycle
            self.history_mgr[actor].control_vector = control_vector
            # let the actor act on its controls and current history
            action_vector, feedback_vector = actor.act(
                self.history_mgr[actor].history, control_vector)
            # update action_vector for current cycle
            self.history_mgr[actor].action_vector = action_vector
            # update feedback_vector for current cycle
            self.history_mgr[actor].feedback_vector = feedback_vector
            # add new action information to decision information matrix of this step
            decision_matrix[actor] = action_vector
            # add new feedback information to feedback information matrix of this step
            response_matrix[actor] = feedback_vector
        # generate percepts
        for actor in self.actors:
            percept_vector = [None] * len(actor.percept_space)
            for influencer in actor.inward_influence_links.keys():
                for p_ch, i_ch, ch_type in actor.inward_influence_links[influencer]:
                    percept_vector[p_ch] = decision_matrix[influencer][i_ch] if ch_type == 'action' else response_matrix[influencer][i_ch]
            # update feedback_vector for current cycle
            self.history_mgr[actor].percept_vector = percept_vector
            percept_matrix[actor] = percept_vector

        self.decision_matrix = decision_matrix
        self.response_matrix = response_matrix
        self.percept_matrix = percept_matrix

        return self

    def learn_cycle(self):
        for actor in self.actors:
            self.history_mgr[actor].record()
            actor.learn(self.history_mgr[actor].history)
        return self

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
            self.register(*actor.inward_control_links.keys())
            # check if the actor can be registered
            self.actors.append(actor)
            self.history_mgr[actor] = helpers.HistoryManager(
                history_maxlen=actor.history_maxlen)

    def deregister(self, actors): pass

    def stats(self): pass


class Actor(abc.ABC):

    def __init__(self, name=None, *args, **kwargs):
        # save a local copy of args and kwargs
        self.args = args
        self.kwargs = kwargs
        # naming the actor to some meaningful way
        self.name = 'actor_{}'.format(id(self)) if name is None else name
        # overridable maximum history length
        self.history_maxlen = self.kwargs.get('history_maxlen', 10)

        self.externally_controlled_channels = set()
        self.messages = []

        # inward signals
        self.control_space = [helpers.Sequence(1)]  # a_in default control
        self.percept_space = []  # <from second person> e_in
        # outward signals
        self.feedback_space = [helpers.Sequence(1)]  # a_out default feedback
        self.action_space = []  # <from second person> e_out

        # list of control and influence links
        # structure:
        #   *_control_links = {actor_1: {(2,2),(1,2),(0,0)}, actor_2: {...}, ...}
        self.inward_control_links = defaultdict(set)
        self.outward_control_links = defaultdict(set)
        # structure:
        #   *_influence_links = {actor_1: {(2,2,'action'),(1,2,'feedback')}, actor_2: {...}, ...}
        self.inward_influence_links = defaultdict(set)
        self.outward_influence_links = defaultdict(set)

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

    def is_controlling(self, actor):
        """
        check through the control chain
        """
        # the actor is in the controlled list
        if actor in self.outward_control_links.keys():
            return True
        # check further down the line
        for controlled_actor in self.outward_control_links.keys():
            if controlled_actor.controlling(actor):
                return True
        # not controlled anywhere in the hierarchy
        return False

    def controlled_by(self, controller, control_channels=[]):
        # the input actor (actor) must not be controlled by the current actor (self)
        # otherwise it will create a cycle in the control mechanism
        if self.is_controlling(controller):
            raise RuntimeError(
                "The actor is already being controlled by the current actor.")
        # request all control channels
        if not control_channels:
            control_channels = range(len(self.control_space))
        # only valid channels are requested
        if not set(control_channels) <= set(range(len(self.control_space))):
            raise RuntimeError("The requested channels are not valid.")
        # all channels are available
        if any(ch in self.externally_controlled_channels for ch in control_channels):
            raise RuntimeError("The requested channel is already allocated.")
        # allot the requested channels
        for c_channel in control_channels:
            # assumption: action-space is added to the controller not selected by the controller
            a_channel = len(controller.action_space)
            controller.action_space.append(self.control_space[c_channel])
            # store inward and outward links
            self.inward_control_links[controller].add((c_channel, a_channel))
            controller.outward_control_links[self].add((a_channel, c_channel))
            # mark the control channel as controlled
            self.externally_controlled_channels.add(c_channel)
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
        if actor in self.inward_control_links.keys():
            # if condition to potentially remove empty list keys in the control list
            channels = [ch for ch, _ in self.inward_control_links[actor]]
        action_channels = [ch for ch in action_channels if ch not in channels]

        # allocate channels to percept_space
        for a_ch in action_channels:
            p_ch = len(self.percept_space)
            self.inward_influence_links[actor].add((p_ch, a_ch, 'action'))
            actor.outward_influence_links[self].add((a_ch, p_ch, 'action'))
            self.percept_space.append(actor.action_space[a_ch])
        for f_ch in feedback_channels:
            p_ch = len(self.percept_space)
            self.inward_influence_links[actor].add((p_ch, f_ch, 'feedback'))
            actor.outward_influence_links[self].add((f_ch, p_ch, 'feedback'))
            self.percept_space.append(actor.feedback_space[f_ch])
        # return `self` so the function calls may be chained
        return self

    def act(self, history, controls):
        # pre_decision_info should "agree" with the input space
        # output space is "received" from another actors
        return [utils.random_sample(a) for a in self.action_space], [utils.random_sample(f) for f in self.feedback_space]

    def state(self, history, pre_decision_info):
        return history[-1]

    def response(self, history, controls):
        return [utils.random_sample(f) for f in self.feedback_space]

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

    rl = Arena()

    agent = Actor('Agent007')
    domain = Actor('Starship')

    domain.control_space = [helpers.Reals(), helpers.Naturals(
    ), helpers.Sequence(10), helpers.Interval(2.0)]
    domain.feedback_space = [
        helpers.Naturals(), helpers.Sequence(2), helpers.Interval(2.0)]

    domain.controlled_by(agent)
    agent.influenced_by(domain)

    rl.register(agent)
    rl.register(domain)

    for n in range(15):
        rl.tick()

    agentA = Actor('AgentA')
    agentB = Actor('AgentB')

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
