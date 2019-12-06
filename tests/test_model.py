from minesrl.agent import BaseAgent, evaluate_agent
from minesrl.envs import MinesEnv


def test_baseline():
    env = MinesEnv(16, 1, 1)
    agent = BaseAgent(env)

    evaluate_agent(agent, epoch=100)
