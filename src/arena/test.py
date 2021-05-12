
import arena

agent = Actor(name='agent')
domain = Actor(name='domain')
domain.control_space = [arena.Reals(), arena.Naturals(),
                        arena.Sequence(10), arena.Interval(2.0)]

domain.controlled_by(agent)

rl = Arena()
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
