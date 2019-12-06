from minesrl.game import *
from minesrl.envs import MinesConsoleEnv


class MinesConsole(MinesGame):
    def __init__(self, row=16, col=16, mine=40):
        env = MinesConsoleEnv(row, col, mine)
        super().__init__(env)

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
