from minesrl.envs import MinesEnv


class MinesConsoleEnv(MinesEnv):
    def render(self):
        map = self.observation_space
        print('   |', end='')
        for c in range(self.cols):
            print('{0:2d}'.format(c % 10), end='')
        print(' ')
        for c in range(self.cols + 2):
            print('--', end='')
        print(' ')
        for r in range(self.rows):
            print('{0:3d}| '.format(r), end='')
            for c in range(self.cols):
                if map[r][c] == self.UNKNOWN:
                    print('.', end=' ')
                elif map[r][c] == self.MINE:
                    print('💣', end=' ')
                elif map[r][c] == 0:
                    print(' ', end=' ')
                else:
                    print(map[r][c], end=' ')
            print()
