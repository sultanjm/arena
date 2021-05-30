import arena
from arena.helpers import Category
import example

import numpy as np

from arena import Continuous, Discrete, Category

space = [
    Category(['up', 'down', 'left', 'right'], 'x'),
    Discrete(4, 7, 'y'),
    Continuous(-np.inf, 3.0, 'z'),
    Continuous(np.inf, -np.inf, 'k'),
    Discrete(np.inf, 5.8),
    Discrete(-np.inf, -9, step=3.16889),
    Discrete(-np.inf, np.inf)
]

x = [a.sample() for a in space]
y = [a.sample() for a in space]

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
