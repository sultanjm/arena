import arena

grl = arena.Arena('grl-arena')

agent = arena.Actor('agent')
domain = arena.MDP('ergodic-mdp')

domain.controlled_by(agent)
agent.influenced_by(domain)

grl.register(domain, agent)

for n in range(20):
    grl.tick()

grl


rl = Arena()

agent = Actor('Agent007')
domain = Actor('Starship')
domain.control_space = [helpers.Interval(
    [0.0, 1.0], 'pos_x'), helpers.Interval([0.0, 1.0], 'pos_y')]
domain.feedback_space = [helpers.Sequence(range(10), 'levels'), helpers.Sequence([
    'low', 'moderate', 'high'], 'status')]

domain.controlled_by(agent)
agent.influenced_by(domain)

rl.register(agent)
rl.register(domain)

for n in range(15):
    rl.tick()

agentA = Actor('AgentA')
agentB = Actor('AgentB')

pd = Actor('PD')
pd.control_space = [helpers.Sequence(
    ['up', 'down']), helpers.Sequence(['left', 'right'])]
pd.feedback_space = pd.control_space

pd.controlled_by(agentA, control_channels=[0])
pd.controlled_by(agentB, control_channels=[1])

agentA.influenced_by(pd, feedback_channels=[1])
agentB.influenced_by(pd, feedback_channels=[0])

rl.register(agentB, pd)
rl.register(pd)
rl.register(agentA)

for _ in range(15):
    rl.tick()

rl
