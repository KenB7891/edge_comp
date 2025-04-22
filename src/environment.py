import random
import simpy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class EdgeEnv(gym.Env):
    """Gymnasium environment for edge‐computing task scheduling."""
    metadata = {"render_modes": []}

    def __init__(self,
                 num_nodes: int = 3,
                 cpu_caps: list = None,
                 max_tasks: int = 100,
                 seed: int = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.cpu_caps = cpu_caps or [1.0] * num_nodes
        self.max_tasks = max_tasks

        # Action: pick a node 0 ... num_nodes-1
        self.action_space = spaces.Discrete(self.num_nodes)

        # Observation: [cpu_req, mem_req, bw_req] + [util_node_0 ... util_node_N]
        obs_dim = 3 + self.num_nodes
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Simulation state
        self.env: simpy.Environment
        self.current_task: dict
        self.tasks_scheduled = 0
        self.np_random = None
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env = simpy.Environment()
        self.tasks_scheduled = 0
        # (optional) start background arrival process here
        return self._next_observation(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Simulate execution: simple model exec_time = cpu_req / cpu_cap
        cpu_req = self.current_task["cpu_req"]
        exec_time = cpu_req / self.cpu_caps[action]

        # Advance the SimPy clock
        event = self.env.timeout(exec_time)
        self.env.run(until=event)

        # Reward = negative latency
        reward = -exec_time
        self.tasks_scheduled += 1

        done = self.tasks_scheduled >= self.max_tasks
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)

        return obs, reward, done, False, {}

    def _next_observation(self):
        # Sample a new task
        cpu_req = self.np_random.uniform(0.1, 1.0)
        mem_req = self.np_random.uniform(0.1, 1.0)
        bw_req  = self.np_random.uniform(0.1, 1.0)
        self.current_task = {"cpu_req": cpu_req, "mem_req": mem_req, "bw_req": bw_req}

        # For now, node utils are zeroed (you can track them in future)
        node_utils = np.zeros(self.num_nodes, dtype=np.float32)

        obs = np.concatenate([[cpu_req, mem_req, bw_req], node_utils]).astype(np.float32)
        return obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass