from minesrl.agent import BaseAgent
import numpy as np
from minesrl.envs import MinesEnv


class QLearningAgent(BaseAgent):
    def __init__(self, env, policy, alpha=0.4, gamma=0.9, epochs=1000):
        super().__init__(env)
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.epochs = epochs

        self.nb_actions = env.rows * env.cols
        self.nb_states = env.rows * env.cols * 11

        self.qtable = np.zeros((self.nb_states, self.nb_actions))

    def save(self, path):
        pass

    def load(self, path):
        pass

    def train(self):
        reward_mean = []
        win_prob = []
        reward_total = 0
        win_total = 0
        for i in range(self.epochs):

            prev_state = self.env.reset()

            while True:
                # Get select action from policy
                prev_state_wrap = self._wrap_state(prev_state)
                print(prev_state_wrap)
                print(self.qtable.shape)
                act = self.policy.get_action(
                    self.qtable[prev_state_wrap, :])

                # Take reward
                next_state, reward, done, info = self.env.step(
                    self._unwrap_action(act))

                self.update_qtable(prev_state_wrap, act,
                                   reward, self._wrap_state(next_state))

                prev_state = next_state

                reward_total += reward
                if done:
                    if info['won']:
                        win_total += 1
                    break

            reward_mean.append(float(reward_total) / (i + 1))
            win_prob.append(float(win_total) / (i + 1))

        return reward_mean, win_prob

    def get_action(self):
        pass

    def update_qtable(self, prev_state, action, reward, next_state):
        qa = np.max(self.qtable[next_state, :])
        self.qtable[prev_state, action] += self.alpha * \
            (reward + self.gamma * qa - self.qtable[prev_state, action])

    def _wrap_state(self, state):
        result = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                c = state[i, j]
                if c == MinesEnv.MINE:
                    c = 9
                if c == MinesEnv.UNKNOWN:
                    c = 10
                result = result * 11 + c

        return result

    def _wrap_action(self, action):
        action = tuple(action)
        return self.env.rows * action[0] + action[1]

    def _unwrap_action(self, action):
        row = action // self.env.rows
        col = action % self.env.rows

        return (row, col)
