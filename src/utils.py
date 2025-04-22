import gymnasium as gym
from src.environment import EdgeEnv

def register_env():
    gym.register(
        id="EdgeScheduling-v0",
        entry_point="src.environment:EdgeEnv",
        max_episode_steps=100,    # or whatever makes sense
    )