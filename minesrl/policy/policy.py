import numpy as np


class Policy(object):
    def get_action(self, **kwargs):
        raise NotImplementedError

    def get_config(self):
        return {}


class EpsGreedPolicy(Policy):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def get_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)

        return action
