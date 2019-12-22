from minesrl.agent import BaseAgent
from minesrl.envs import MinesEnv
from minesrl.policy import EpsGreedPolicy, GreedyQPolicy
from minesrl.model import DQN
from minesrl.memory import SequentialMemory

import numpy as np
from keras.optimizers import Adam
import keras.backend as K


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class QLearningAgent(BaseAgent):

    def __init__(self, env, policy=None, test_policy=None, model=None, memory=None, optimizer=None, metrics=[]):
        super().__init__(env)

        if policy == None:
            policy = EpsGreedPolicy()
        self.policy = policy

        if test_policy == None:
            test_policy = GreedyQPolicy()
        self.test_policy = test_policy

        if memory == None:
            memory = SequentialMemory(10000)
        self.memory = memory

        if model == None:
            row = env.rows
            col = env.cols
            model = DQN((row, col), row * col)
        self.model = model

        if optimizer == None:
            optimizer = Adam(lr=1e-4)
        self.optimizer = optimizer

        self.metrics = metrics
        self.metrics += [mean_q]

    def get_action(self, state):
        q_val = self.model.call(state)
        return self.test_policy.get_action(q_val)
