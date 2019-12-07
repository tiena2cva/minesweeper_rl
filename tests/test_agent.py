from minesrl.agent import *
from minesrl.envs import MinesEnv


def test_baseline():
    env = MinesEnv(16, 1, 1)
    agent = BaseAgent(env)

    evaluate_agent(agent, epoch=100)


def test_PPOagent():
    env = MinesEnv(8, 8, 8)
    agent = PPOAgent(env)
    agent.train()

    evaluate_agent(agent, epoch=100)
