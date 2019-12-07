from minesrl.game import *
from minesrl.envs import MinesConsoleEnv


class MinesConsole(MinesGame):
    def __init__(self, row=16, col=16, mine=40):
        env = MinesConsoleEnv(row, col, mine)
        super().__init__(env)

    def render(self):
        map = self.env.get_obs()
        print('   |', end='')
        for c in range(self.env.cols):
            print('{0:2d}'.format(c % 10), end='')
        print(' ')
        for c in range(self.env.cols + 2):
            print('--', end='')
        print(' ')
        for r in range(self.env.rows):
            print('{0:3d}| '.format(r), end='')
            for c in range(self.env.cols):
                if map[r][c] == self.env.UNKNOWN:
                    print('.', end=' ')
                elif map[r][c] == self.env.MINE:
                    print('💣', end=' ')
                elif map[r][c] == 0:
                    print(' ', end=' ')
                else:
                    print(map[r][c], end=' ')
            print()

    def get_input(self):
        inp = input('Action: ')
        try:
            inps = inp.split()
            r = int(inps[0])
            c = int(inps[1])

            return r, c
        except Exception:
            return inp

    def render_done(self, won, reward):
        self.render()
        if won:
            print('You won!!! With reward = ' + str(reward))
        else:
            print('You lose!!! But get reward = ' + str(reward))
