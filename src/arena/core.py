from . import helpers

from collections import defaultdict
import numpy as np


class Arena:
    """The main class of the framework. 

    The core object connecting all actors in the system.

    Attributes:
        name (str): A str identifer of the arena, by default ``arena:id(self)``.
        actors (list): A list of actors registered in the arena.
        history_mgr (dict of HistroyManager: Actor): A dict of history managers for each actor, keyed by actors.
        steps (int): An int counting the number of steps ticks.

    Example:
        >>> arena = Arena()
        >>> actor = Actor()
        >>> arena.register(actor)
        >>> arena.tick()
    """

    def __init__(self, name=None, *args, **kwargs):
        # name the arena
        self.name = 'arena:{}'.format(id(self)) if name is None else name
        # list of actors in order
        self.actors = list()
        # list of history managers for every actor
        self.history_mgr = defaultdict(helpers.HistoryManager)
        # interaction steps
        self.steps = 0

    def tick(self, steps=1):
        """
        performs a single interaction cycle
        loops through the registered actors
        provides relevent controls to the actors
        prepare percepts for them
        let every actor learn from history
        """
        for step in range(steps):
            # start new action, feedback, percept, and evaluation matrices for the current step
            # structure:
            #   action_matrix[actor] = [action vector from action_space]
            #   feedback_matrix[actor] = [feedback vector from feedback_space]
            #   percept_matrix[actor]  = [action & feedback vector from percept_space]
            #   evaluation_matrix[actor]  = [evaluation vector from percept_space]
            action_matrix = defaultdict(list)
            control_matrix = defaultdict(list)
            feedback_matrix = defaultdict(list)
            percept_matrix = defaultdict(list)
            evaluation_matrix = defaultdict(list)
            reward_matrix = defaultdict(list)
            # go through each actor
            # assuming the actors are in order
            for actor in self.actors:
                # generate a "random" control vector just in case nobody is controling
                control_vector = [c.random_sample()
                                  for c in actor.control_space]
                # check the links
                # if the actor is being controlled
                for controller in actor.inward_control_links.keys():
                    # there is a controller
                    # the links are stored as control-action channel pairs
                    for c_ch, a_ch in actor.inward_control_links[controller]:
                        # get the action from controller to the actor
                        control_vector[c_ch] = action_matrix[controller][a_ch]
                control_matrix[actor] = control_vector
                # let the actor act on its controls and current history
                action_vector = actor.act(
                    self.history_mgr[actor].history, control_vector)
                # get the feedback vector from the actor
                feedback_vector = actor.respond(
                    self.history_mgr[actor].history, control_vector, action_vector)
                # ask for evaluation from the actor
                evaluation_vector = actor.evaluate(
                    self.history_mgr[actor].history, control_vector, action_vector, feedback_vector)

                # add new action information to decision information matrix of this step
                action_matrix[actor] = action_vector
                # add new feedback information to feedback information matrix of this step
                feedback_matrix[actor] = feedback_vector
                evaluation_matrix[actor] = evaluation_vector
            # generate percepts
            # learn from current iteration using the information
            for actor in self.actors:
                percept_vector = [p.random_sample()
                                  for p in actor.percept_space]
                for influencer in actor.inward_influence_links.keys():
                    for p_ch, i_ch, ch_type in actor.inward_influence_links[influencer]:
                        percept_vector[p_ch] = action_matrix[influencer][i_ch] if ch_type == 'action' else feedback_matrix[influencer][i_ch]
                # update feedback_vector for current cycle
                self.history_mgr[actor].percept_vector = percept_vector
                percept_matrix[actor] = percept_vector
                # ask for
                reward_vector = np.zeros(len(actor.action_space))
                for controlled in actor.outward_control_links.keys():
                    for a_ch, c_ch in actor.outward_control_links[controlled]:
                        reward_vector[a_ch] = evaluation_matrix[controlled][c_ch]
                reward_matrix[actor] = reward_vector
                # learn from the interation at the history
                next_state = actor.learn(
                    self.history_mgr[actor].history,
                    control_matrix[actor],
                    action_matrix[actor],
                    feedback_matrix[actor],
                    percept_matrix[actor],
                    reward_matrix[actor]
                )
                # record the current interaction
                self.history_mgr[actor].record(
                    control_matrix[actor],
                    action_matrix[actor],
                    feedback_matrix[actor],
                    percept_matrix[actor]
                )
                # update the state
                self.history_mgr[actor].history.state = next_state
            # done with the current step
            self.steps += 1
        # let's return `self` to allow chained method calls
        return self

    def register(self, *actors):
        """
        register actors to arena
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
            # state of empty history
            self.history_mgr[actor].history.state = actor.reset()

    def deregister(self, actors):
        # TODO: not yet implemented
        pass

    def stats(self):
        # TODO: implement a standard interface for stats
        pass


class Actor:
    """
    actor of the system
    """

    def __init__(self, name=None, history_maxlen=10, *args, **kwargs):
        # naming the actor to some meaningful way
        self.name = 'actor:{}'.format(id(self)) if name is None else name
        # overridable maximum history length
        self.history_maxlen = history_maxlen

        self.externally_controlled_channels = set()
        self.messages = list()

        # inward signals
        self.control_space = kwargs.get(
            'control_space', list())  # a_in default control
        self.percept_space = list()  # <from second person> e_in
        # outward signals
        self.feedback_space = kwargs.get(
            'feedback_space', list())  # a_out default feedback
        self.action_space = list()  # <from second person> e_out
        # internal signal
        # internal states of the actor
        self.state_space = kwargs.get(
            'state_space', list())

        # list of control and influence links
        # structure:
        #   *_control_links = {actor_1: [(2,2),(1,2),(0,0), ...], actor_2: ...}
        self.inward_control_links = defaultdict(set)
        self.outward_control_links = defaultdict(set)
        # structure:
        #   *_influence_links = {actor_1: [(2,2,'action'),(1,2,'feedback'), ...], actor_2: ...}
        self.inward_influence_links = defaultdict(set)
        self.outward_influence_links = defaultdict(set)
        # user-constructor
        self.setup(*args, **kwargs)

    def internal_state(self, history):
        """
        structural assumption:
        the state signal is inside the history
        a history update tuple is:
        (controls + actions + feedbacks + percepts + state)
        """
        if not history.steps:
            # initiated at the empty history
            # start with an initial state
            return self.reset()
        last_step = history[-1]
        # last_step = controls + actions + feedbacks + percepts + state
        offset = len(self.control_space) + len(self.action_space) + \
            len(self.feedback_space) + len(self.percept_space)
        return last_step[offset:offset + len(self.state_space)]

    # history has to have a `state`, be it the (only) default state
    def state(self, history, controls, actions, feedbacks, percepts):
        if not history.steps and not history.state:
            history.state = self.reset()

    def reset(self):
        # initial internal state of the actor
        return [s.random_sample() for s in self.state_space]

    def says(self, message):
        # messages should be updated on each 'tick'
        return True if self.messages.contains(message) else False

    def controlling(self, actor):
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
        """
        actor is being controlled by the controller
        """
        # the input actor (actor) must not be controlled by the current actor (self)
        # otherwise it will create a cycle in the control mechanism
        if self.controlling(controller):
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
        for c_ch in control_channels:
            # assumption: action-space is added to the controller not selected by the controller
            a_ch = len(controller.action_space)
            controller.action_space.append(self.control_space[c_ch])
            # store inward and outward links
            self.inward_control_links[controller].add((c_ch, a_ch))
            controller.outward_control_links[self].add((a_ch, c_ch))
            # mark the control channel as controlled
            self.externally_controlled_channels.add(c_ch)
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

    def setup(self, *args, **kwargs):
        # set percept_space and action_space for the actors being controlled from it
        # environment -> agent
        # agent.action_space = env.control_space
        # agent.percept_space = env.feedback_space
        # evn receives (control_space, percept_space) at input, (a, None)
        # env produces (feedback_space, action_space) at output, (e, None)
        return self

    def act(self, history, controls):
        # pre_decision_info should "agree" with the input space
        # output space is "received" from another actors
        return [a.random_sample() for a in self.action_space]

    def respond(self, history, controls, actions):
        return [f.random_sample() for f in self.feedback_space]

    def evaluate(self, history, controls, actions, feedbacks):
        # the size of returned evaluation should match pre_decision_info
        # the return vector should be of the size of the control channels
        # takes in complete history -> R^[size of controls]
        return np.zeros(len(self.control_space))

    def learn(self, history, controls, actions, feedbacks, percepts, rewards):
        # learn nothing, keep resetting
        return self.reset()

    def render(self):
        pass
