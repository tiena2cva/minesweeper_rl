from envs.mines_env import MinesEnv


class MinesGame(object):
    def __init__(self, row=16, col=16, mine=40):
        self.env = MinesEnv(row=row, col=col, mine=mine)

    def render(self):
        raise NotImplementedError

    def get_input(self):
        raise NotImplementedError

    def step(self, row, col):
        return self.env.step((row, col))

    def render_done(self, won, reward):
        pass

    def play(self):
        done = False
        reward = 0
        while not done:
            self.render()
            inp = self.get_input()

            if inp is str and inp.lower() == 'exit':
                break

            row, col = inp
            assert row is int and col is int

            _, r, done, won = self.step(row, col)
            reward += r

            if done:
                self.render_done(won, reward)
