from minesrl.agent import QLearningAgent
from minesrl.policy import EpsGreedPolicy
from minesrl.envs import MinesEnv, MinesEnvWrapper
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

if __name__ == "__main__":
    env = MinesEnvWrapper(row=8, col=8, mine=8)
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    print(history.history)
    history = dqn.test(env, nb_episodes=5, visualize=False)
    print(history.history)
    """
    reward_mean, win_prob = [], []

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('prob %', color=color)
    line_win = ax1.plot(win_prob, color=color, label='Win prob')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('points', color=color)
    line_reward = ax2.plot(reward_mean, color=color, label='Reward mean')
    ax2.tick_params(axis='y', labelcolor=color)

    lns = line_win + line_reward
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    """
