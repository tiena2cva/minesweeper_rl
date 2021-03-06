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
        self.obs = np.full(
            [self.rows, self.cols], MinesEnv.UNKNOWN)

        # OpenAI gym param
        self.action_space = spaces.MultiDiscrete(
            [self.rows, self.cols])
        self.observation_space = spaces.Box(
            low=MinesEnv.MINE, high=8, shape=(self.rows, self.cols))
        self.seed()

        self.__init_game()

    def get_obs(self):
        return np.copy(self.obs)

    # OpenAI gym API
    def reset(self):
        """Resets the state of the environment and returns an initial
        observation.

        Returns:
            observation (object): the initial observation.
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

        self.obs = np.full(
            [self.rows, self.cols], MinesEnv.UNKNOWN)

        # self.step(self.start_pos)
        return self.get_obs()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done,
        info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current
                environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further
                step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)
        """

        reward = 0
        done = False
        won = False
        coord = tuple(action)
        if self.obs[coord] != MinesEnv.UNKNOWN:
            # Clicked on an opened cell
            reward = -2
        elif self.map[coord] == MinesEnv.MINE:
            # Clicked on a mine!
            self.obs[coord] = MinesEnv.MINE
            reward = -99        # you lose
            done = True
        else:
            # Check if it clicks on a random cell with unknown neighbors
            reward = -1
            for n in self.__neighbors[coord]:
                if self.obs[n] != MinesEnv.UNKNOWN:
                    reward = 0
                    break

            count = self.__open_cell(coord)
            self.coords_to_clear -= count
            if self.coords_to_clear <= 0:
                reward = 100     # you won.
                done = True
                won = True
            else:
                if reward != -1:
                    reward = count

        return (self.get_obs(), reward, done, {'won': won})

    def render(self, mode='human'):
        """Renders the environment.
        """
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
        if self.obs[coord] != MinesEnv.UNKNOWN:
            return 0

        self.obs[coord] = self.map[coord]
        count = 1
        if self.map[coord] == 0:
            for n in self.__neighbors[coord]:
                count += self.__open_cell(n)
        return count
