from gym import Wrapper, Env, spaces
import numpy as np
import random


class MinesEnv(Env):
    UNKNOWN = -1
    MINE = -2

    def __init__(self, row=16, col=16, mine=40):
        self.rows = row
        self.cols = col
        self.mines = mine

        self.coords_to_clear = self.rows * self.cols - self.mines
        self.map = np.full((self.rows, self.cols), MinesEnv.UNKNOWN)
        self.__neighbors = np.zeros((self.rows, self.cols), dtype=object)
        self.start_pos = (self.rows // 2, self.cols // 2)

        # OpenAI gym param
        self.action_space = spaces.Tuple(
            (spaces.Discrete(row), spaces.Discrete(col)))
        self.observation_space = np.full(
            [self.rows, self.cols], MinesEnv.UNKNOWN)

        self.seed()

        self.__init_game()

    # OpenAI gym API
    def reset(self):
        """ Reset all attributes to init state
        """
        self.map = np.full([self.rows, self.cols], MinesEnv.UNKNOWN)
        self.coords_to_clear = self.rows * self.cols - self.mines

        # Gen random mine in the map
        no_mines = self.mines
        while no_mines > 0:
            r = random.randrange(self.rows)
            c = random.randrange(self.cols)

            if (r, c) != self.start_pos and self.map[r][c] != MinesEnv.MINE:
                self.map[r][c] = MinesEnv.MINE
                no_mines -= 1

        # Calculate cell's value for fast lookups
        for col in range(self.cols):
            for row in range(self.rows):
                if self.map[row][col] != MinesEnv.MINE:
                    no_mines = self.__count_mines(row, col)
                    if no_mines > 0:
                        self.map[row][col] = no_mines
                    else:
                        self.map[row][col] = 0

        self.observation_space = np.full(
            [self.rows, self.cols], MinesEnv.UNKNOWN)

        self.step(self.start_pos)
        return self.observation_space

    def step(self, coord):
        reward = 0
        done = False
        if self.observation_space[coord] != MinesEnv.UNKNOWN:
            reward = -2
        elif self.map[coord] == MinesEnv.MINE:
            # Clicked on a mine!
            self.observation_space[coord] = MinesEnv.MINE
            reward = -99
            done = True
        else:
            count = self.__open_cell(coord)
            self.coords_to_clear -= count
            if self.coords_to_clear <= 0:
                reward = 100     # Yay you won.
                done = True
            else:
                reward = count
        return (self.observation_space, reward, done, None)

    def render(self, mode='human'):
        pass

    # Private function
    def __init_game(self):
        self.__compute_neighbors()

        self.reset()

    def __count_mines(self, row, col):
        """ Count amount of mines adjacent to a cell.
        """
        mines = 0
        neighbor = self.__neighbors[row, col]
        for n in neighbor:
            if self.map[n] == MinesEnv.MINE:
                mines += 1
        return mines

    def __compute_neighbors(self):
        """ Computes the neighbor matrix for quick lookups"""

        for row in range(self.rows):
            for col in range(self.cols):
                self.__neighbors[row][col] = self.__find_neighbors(row, col)

    def __find_neighbors(self, row, col):
        """ Returns list of neighbors of this cell
        """
        if not ((-1 < row < self.rows) and (-1 < col < self.cols)):
            return []

        neighbors = []
        for c in range(col - 1, col + 2):
            for r in range(row - 1, row + 2):
                if ((row != r or col != c) and
                    (0 <= c < self.cols) and
                        (0 <= r < self.rows)):
                    neighbors.append((r, c))

        return neighbors

    def __open_cell(self, coord):
        if self.map[coord] == MinesEnv.MINE:
            return 0
        if self.observation_space[coord] != MinesEnv.UNKNOWN:
            return 0

        self.observation_space[coord] = self.map[coord]
        count = 1
        if self.map[coord] == 0:
            for n in self.__neighbors[coord]:
                count += self.__open_cell(n)
        return count


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

            if env.observation_space[row][col] != MinesEnv.UNKNOWN:
                assert env.map[row][col] == env.observation_space[row][col]
                if env.observation_space[row][col] != MinesEnv.MINE:
                    knew += 1

    # Test coords_to_clear
    assert env.coords_to_clear == env.rows * env.cols - knew - env.mines


def __test_env(env):
    __test_action_space(env)
    __test_map(env)

    action = env.action_space.sample()
    env.step(action)
    __test_map(env)


if __name__ == "__main__":
    mines = MinesEnv()
    for _ in range(100):
        __test_env(mines)

    print('Everything passed')
