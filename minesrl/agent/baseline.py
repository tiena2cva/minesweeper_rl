import matplotlib.pyplot as plt
from minesrl.envs import MinesEnv


class BaseAgent(object):
    def __init__(self, env):
        super().__init__()

        self.env = env

    def save(self, path):
        pass

    def load(self, path):
        pass

    def train(self):
        pass

    def get_action(self):
        act = self.env.action_space.sample()
        return act

    def play(self):
        self.env.reset()

        done = False
        reward = 0
        while not done:

            action = self.get_action()

            _, r, done, info = self.env.step(action)
            reward += r

        return reward, info


def evaluate_agent(agent, epoch=10000, plot=True):
    count_win = 0
    sum_reward = 0

    win_prob = []
    reward_mean = []
    for i in range(epoch):
        reward, info = agent.play()

        if info['won']:
            count_win += 1

        sum_reward += reward

        win_prob.append(count_win / (i + 1))
        reward_mean.append(sum_reward / (i + 1))

    if plot:
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('prob %', color=color)
        ax1.plot(win_prob, color=color, label='Win prob')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        # we already handled the x-label with ax1
        ax2.set_ylabel('points', color=color)
        ax2.plot(reward_mean, color=color, label='Reward mean')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.show()
