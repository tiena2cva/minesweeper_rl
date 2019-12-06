class BaseAgent(object):
    def __init__(self, env):
        super().__init__()

        self.env = env

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def play(self):
        self.env.reset()
