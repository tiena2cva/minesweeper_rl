from minesrl.agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, env, model, policy):
        super().__init__(env)
        self.policy = policy

    def save(self, path):
        pass

    def load(self, path):
        pass

    def train(self):
        pass

    def get_action(self):
        pass
