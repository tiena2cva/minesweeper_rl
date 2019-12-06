from minesrl.envs import *


class MinesGame(object):
    def __init__(self, env):
        self.env = env

    def render(self):
        self.env.render()

    def step(self, row, col):
        return self.env.step((row, col))

    def get_input(self):
        raise NotImplementedError

    def render_done(self, won, reward):
        raise NotImplementedError

    def play(self):
        done = False
        reward = 0
        while not done:
            self.render()
            inp = self.get_input()

            if type(inp) == str and inp.lower() == 'exit':
                break

            row, col = inp

            _, r, done, info = self.step(row, col)
            reward += r

            if done:
                self.render_done(info['won'], reward)
