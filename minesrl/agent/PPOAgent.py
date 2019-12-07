from minesrl.agent import BaseAgent

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


class PPOAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def train(self):
        env = DummyVecEnv([lambda: self.env])

        model = PPO2(MlpPolicy, env, verbose=1)
        # Train the agent
        model.learn(total_timesteps=10000)
