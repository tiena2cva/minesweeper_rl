import numpy as np
from minesrl.envs import MinesEnvWrapper
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = MinesEnvWrapper(row=10, col=10, mine=10)

np.random.seed(123)
env.seed(123)


def nn_model():
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(100))

    return model


model = nn_model()
model.summary()


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=100, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-3, policy=policy)
dqn.compile(Adam(lr=1e-4), metrics=['mae'])

history = dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)

print(history.history)
