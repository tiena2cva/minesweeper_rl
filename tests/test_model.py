from minesrl.agent import BaseAgent, evaluate_agent
from minesrl.envs import MinesEnv


def test_baseline():
    env = MinesEnv(8, 8, 1)
    agent = BaseAgent(env)

    evaluate_agent(agent)
