import arena
from arena.helpers import Sequence

grl = arena.Arena('grl-arena')

agent = arena.Actor('agent')

mdp = arena.MDP(
    name='ergodic-mdp',
    control_space=[arena.Interval([-1.0, 1.0]), arena.Interval([-1.0, 1.0])],
    state_space=[arena.Sequence(range(4))]
)

pomdp = arena.POMDP(
    name='ergodic-pomdp',
    control_space=[arena.Interval([-1.0, 1.0]), arena.Interval([-1.0, 1.0])],
    state_space=[arena.Sequence(range(4))],
    feedback_space=[arena.Sequence(range(6))]
)

mdp.controlled_by(agent)
agent.influenced_by(mdp)

grl.register(mdp, agent)

grl.tick(100)

grl
