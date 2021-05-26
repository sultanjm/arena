import arena

grl = arena.Arena('grl-arena')

agent = arena.Actor('agent')
domain = arena.MDP(
    name='ergodic-mdp',
    control_space=[arena.Interval([-1.0, 1.0]), arena.Interval([-1.0, 1.0])],
    state_space=[arena.Sequence(range(4))]
)

domain.controlled_by(agent)
agent.influenced_by(domain)

grl.register(domain, agent)

grl.tick(100)

grl
