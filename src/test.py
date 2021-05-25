import arena

grl = arena.Arena('grl-arena')

agent = arena.Actor('agent')
domain = arena.MDP('ergodic-mdp')

control_space = [arena.Interval([-1.0, 1.0]), arena.Interval([-1.0, 1.0])]
feedback_space = [arena.Sequence(range(4))]


def transition_func(self, state, controls, actions):
    pass


domain2 = arena.MDP(
    name='ergodic-mdp',
    control_space=control_space,
    feedback_space=feedback_space,
    transition_func=transition_func
)

domain.controlled_by(agent)
# agent.influenced_by(domain)

grl.register(domain, agent)

grl.tick(1000)
