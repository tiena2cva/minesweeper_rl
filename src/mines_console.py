from mines_core import MinesGame


class MinesConsole(MinesGame):
    def __init__(self, row=16, col=16, mine=40):
        super().__init__(row, col, mine)

    def render(self):
        map = self.env.observation_space
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


if __name__ == "__main__":
    game = MinesConsole(8, 8, 3)
    game.play()
