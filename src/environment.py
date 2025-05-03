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
                mem_caps: list = None,
                cpu_req_range = (0.1, 1.0),
                mem_req_range = (0.1, 1.0),
                arrival_rate: float = 1.0,
                max_tasks: int = 100,
                seed: int = None):

        super().__init__()
        self.num_nodes = len(cpu_caps)
        self.cpu_caps = cpu_caps or [1.0] * num_nodes
        self.mem_caps = mem_caps or [1.0] * num_nodes
        self.cpu_req_min, self.cpu_req_max = cpu_req_range
        self.mem_req_min, self.mem_req_max = mem_req_range
        self.arrival_rate = arrival_rate
        self.max_tasks = max_tasks

        # Action: pick a node 0 ... num_nodes-1
        self.action_space = spaces.Discrete(self.num_nodes)

        # Observation: [cpu_req] + [profile] + [util_node_0 ... util_node_N]
        obs_dim = 1 + 4 + self.num_nodes
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
        self.cpu_resources = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_nodes)]
        self.mem_containers = [simpy.Container(self.env, init=cap, capacity=cap) for cap in self.mem_caps]
        self.tasks_scheduled = 0
        # prepare first observation
        obs = self._next_observation()

        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        cpu_req = self.current_task['cpu_req']
        mem_req = self.current_task['mem_req']

        # memory allocation (blocking until enough memory)
        mem_start_time = self.env.now
        mem_evt = self.mem_containers[action].get(mem_req)
        self.env.run(until=mem_evt)
        mem_wait = self.env.now - mem_start_time

        # CPU queueing
        cpu_start_time = self.env.now
        cpu_req_evt = self.cpu_resources[action].request()
        self.env.run(until=cpu_req_evt)
        cpu_wait = self.env.now - cpu_start_time

        # execution
        exec_start_time = self.env.now
        exec_duration = cpu_req / self.cpu_caps[action]
        exec_evt = self.env.timeout(exec_duration)
        self.env.run(until=exec_evt)
        exec_time = self.env.now - exec_start_time

        # release the resources
        self.cpu_resources[action].release(cpu_req_evt)
        self.mem_containers[action].put(mem_req)

        # calculate reward
        latency = mem_wait + cpu_wait + exec_time
        reward = -latency
        self.tasks_scheduled += 1
        done = self.tasks_scheduled >= self.max_tasks

        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)

        return obs, reward, done, False, {}

    def _next_observation(self):
        # Simulate Poisson interarrival
        inter_arrival = self.np_random.exponential(1.0 / self.arrival_rate)
        self.env.run(until=self.env.now + inter_arrival)

         # Sample task requirements
        cpu_req = self.np_random.uniform(self.cpu_req_min, self.cpu_req_max)
        mem_req = self.np_random.uniform(self.mem_req_min, self.mem_req_max)

        cpu_threshold = 0.5
        mem_thresh = 0.5

        # profile encoding:
        # 0 = light CPU, light MEM
        # 1 = heavy CPU, light MEM
        # 2 = light CPU, heavy MEM
        # 3 = heavy CPU, heavy MEM
        if cpu_req <= cpu_threshold:
            if mem_req <= mem_thresh:
                profile = 0
            else:
                profile = 2
        else:
            if mem_req <= mem_thresh:
                profile = 1
            else:
                profile = 3

        self.current_task = {"cpu_req": cpu_req, "mem_req": mem_req, "profile": profile}

        profile_vec = np.zeros(4, dtype=np.float32)
        profile_vec[profile] = 1
        

        # For now, node utils are zeroed (you can track them in future)
        node_utils = np.zeros(self.num_nodes, dtype=np.float32)

        obs = np.concatenate([[cpu_req], profile_vec, node_utils]).astype(np.float32)
        return obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass