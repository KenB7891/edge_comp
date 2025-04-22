import yaml
import torch
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from src.environment import EdgeEnv

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path: str):
    # 1) Load hyperparams
    cfg = load_config(config_path)

    # 2) Register and create env
    gym.register(
        id="EdgeScheduling-v0",
        entry_point="src.environment:EdgeEnv",
        max_episode_steps=cfg.get("max_episode_steps", 100),
    )
    
    # 2a) Build Monitor‑wrapped env factories
    def make_env(rank):
        return lambda: Monitor(
            gym.make(
                "EdgeScheduling-v0",
                num_nodes=3,
                seed=cfg["seed"] + rank
            )
        )

    # 2b) Create parallel envs
    n_envs = 1
    base_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    env = VecMonitor(base_env)

    # 2c) Normalize obs & rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 3) Seed torch for reproducibility
    torch.manual_seed(cfg["seed"])

    # 4) Create a checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000, 
        save_path="./models/", 
        name_prefix="ppo_edge"
    )

    # 5) Instantiate and train PPO
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=cfg["seed"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        n_steps=cfg["n_steps"],
        gamma=cfg["gamma"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        verbose=1,
        tensorboard_log="./logs/",
    )
    model.learn(total_timesteps=cfg["total_timesteps"],
                callback=checkpoint_cb)
    model.save("./models/ppo_edge_final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/agent.yaml",
        help="Path to agent config file"
    )
    args = parser.parse_args()
    main(args.config)