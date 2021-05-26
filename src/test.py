import arena
import example

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

blindmaze = example.BlindMaze()

blindmaze.controlled_by(agent)
agent.influenced_by(blindmaze)

grl.register(blindmaze, agent)

grl.tick(100)

grl
