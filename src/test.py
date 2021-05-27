import arena
import example

import numpy as np

space = arena.Space([0, 1, 2], range(4), (0.0, 1.0, 1e-1),
                    (-2.0, -3.0, 1e-1), labels=['x', 'y', 'z', 'k'], name='outer-space')
x = space.random_sample()
y = space.label(x)
grl = arena.Arena('grl-arena')

agent = arena.Actor('agent')

mdp = arena.MDP(
    name='ergodic-mdp',
    control_space=[arena.Sequence(range(2))],
    state_space=[arena.Sequence(range(4))]
)

pomdp = arena.POMDP(
    name='ergodic-pomdp',
    control_space=[arena.Sequence(range(2))],
    state_space=[arena.Sequence(range(4))],
    feedback_space=[arena.Sequence(range(6))]
)

q_agent = example.GreedyQAgent()
blindmaze = example.BlindMaze()

blindmaze.controlled_by(q_agent)
q_agent.influenced_by(blindmaze)

grl.register(blindmaze, q_agent)

grl.tick(5000)

grl
