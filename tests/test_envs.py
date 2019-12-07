from minesrl.envs import *

# For test


def __test_action_space(env):
    action = env.action_space.sample()
    assert len(action) == 2
    assert action[0] >= 0 and action[0] < env.rows
    assert action[1] >= 0 and action[1] < env.cols


def __test_map(env):

    knew = 0
    for col in range(env.cols):
        for row in range(env.rows):
            if env.map[row][col] != MinesEnv.MINE:
                mines = 0
                for c in range(col-1, col+2):
                    for r in range(row-1, row+2):
                        if ((r != row or c != col) and
                            (0 <= c < env.cols) and
                            (0 <= r < env.rows) and
                                env.map[r][c] == MinesEnv.MINE):
                            mines += 1
                assert env.map[row][col] == mines

            if env.obs[row][col] != MinesEnv.UNKNOWN:
                assert env.map[row][col] == env.obs[row][col]
                if env.obs[row][col] != MinesEnv.MINE:
                    knew += 1

    # Test coords_to_clear
    assert env.coords_to_clear == env.rows * env.cols - knew - env.mines


def __test_env(env):
    __test_action_space(env)
    __test_map(env)

    action = env.action_space.sample()
    env.step(action)
    __test_map(env)


def test_envs():
    mines = MinesEnv()
    for _ in range(100):
        __test_env(mines)
