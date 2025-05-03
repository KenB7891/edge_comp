import numpy as np
from stable_baselines3 import PPO
from src.environment import EdgeEnv

# Configuration
SCENARIOS = {
    'random': {'cpu_range': (0.1, 1.0), 'mem_range': (0.1, 1.0)},
    'heavy': {'cpu_range': (0.6, 1.0), 'mem_range': (0.6, 1.0)},
    'light': {'cpu_range': (0.1, 0.4), 'mem_range': (0.1, 0.4)},}
DEFAULT_SEEDS = [0, 10, 42, 100, 500] # fix once for all schedulers
NUM_TASKS = 200     # same max_tasks as in training
CPU_CAPS = [0.25, 0.5, 1.0, 2.0]
MEM_CAPS = [2.0, 4.0, 8.0, 16.0]

# trained model
def run_rl(seed, ranges):
    # 1) Instantiate env with fixed seed
    env = EdgeEnv(num_nodes=3,
                          cpu_caps=CPU_CAPS,
                          mem_caps=MEM_CAPS,
                          seed=seed,
                          max_tasks=NUM_TASKS,
                          cpu_req_range=ranges['cpu_range'],
                          mem_req_range=ranges['mem_range'])
    obs, _ = env.reset()
    # 2) Load trained model (make sure path is correct)
    model = PPO.load("models/ppo_edge_final.zip")
    total_latency = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_latency += -reward
    return total_latency

# round-robin
def run_rr(seed, ranges):
    env = EdgeEnv(num_nodes=3,
                          cpu_caps=CPU_CAPS,
                          mem_caps=MEM_CAPS,
                          seed=seed,
                          max_tasks=NUM_TASKS,
                          cpu_req_range=ranges['cpu_range'],
                          mem_req_range=ranges['mem_range'])
    obs, _ = env.reset()
    total_latency = 0.0
    done = False
    idx = 0
    while not done:
        # round-robin: cycle through nodes 0,1,2,0,1...
        action = idx % env.num_nodes
        obs, reward, done, truncated, _ = env.step(action)
        total_latency += -reward
        idx += 1
    return total_latency

# first-come first-served
def run_fcfs(task_list, cpu_caps):
    num_nodes = len(cpu_caps)
    next_free = [0.0] * num_nodes
    total_latency = 0.0
    for cpu_req, mem_req, profile in task_list:
        # Compute execution time per node
        exec_times = [cpu_req / cap for cap in cpu_caps]
        # Assign task to the node that becomes free first
        i = int(np.argmin(next_free))
        start = next_free[i]
        finish = start + exec_times[i]
        next_free[i] = finish
        total_latency += finish
    return total_latency

# Random task assignment
def run_random(task_list, cpu_caps, seed):
    num_nodes = len(cpu_caps)
    next_free = [0.0] * num_nodes
    total_latency = 0.0
    rng = np.random.default_rng(seed)
    for cpu_req, mem_req, profile in task_list:
        exec_times = [cpu_req / cap for cap in cpu_caps]
        i = int(rng.integers(num_nodes))
        start = next_free[i]
        finish = start + exec_times[i]
        next_free[i] = finish
        total_latency += finish
    return total_latency

def print_stats(name, data):
    mean = np.mean(data)
    std = np.std(data)
    avg_per_task = mean / NUM_TASKS
    print(f"{name}: mean={mean:.1f}, std={std:.1f}, avg_per_task={avg_per_task:.3f}")
# Main evaluation: generate task list once, then run schedulers
def main():
    # Storage for latencies across seeds
    for scenario_name, ranges in SCENARIOS.items():
    
        rl_lats = []
        rr_lats = []
        fcfs_lats = []
        rand_lats = []

        for seed in DEFAULT_SEEDS:
            # Generate identical task list
            env = EdgeEnv(num_nodes=3,
                          cpu_caps=CPU_CAPS,
                          mem_caps=MEM_CAPS,
                          seed=seed,
                          max_tasks=NUM_TASKS,
                          cpu_req_range=ranges['cpu_range'],
                          mem_req_range=ranges['mem_range'])
            task_list = []
            obs, _ = env.reset()
            done = False
            while not done:
                t = env.current_task
                task_list.append((t['cpu_req'], t['mem_req'], t['profile']))
                obs, reward, done, truncated, _ = env.step(0)  # dummy action to advance

            cpu_caps = env.cpu_caps

            # Run each scheduler
            rl_lats.append(run_rl(seed, ranges))
            rr_lats.append(run_rr(seed, ranges))
            fcfs_lats.append(run_fcfs(task_list, cpu_caps))
            rand_lats.append(run_random(task_list, cpu_caps, seed))
        print(f'\n----- Scenario: {scenario_name} -----')
        print(f'\nResults over seeds {DEFAULT_SEEDS} and {NUM_TASKS} tasks each:')
        print_stats("PPO", rl_lats)
        print_stats("Round-Robin", rr_lats)
        print_stats("FCFS", fcfs_lats)
        print_stats("Random", rand_lats)

if __name__ == "__main__":
    main()

'''
# first-come first-served
def run_fcfs():
    env = EdgeEnv(num_nodes=3, cpu_caps=CPU_CAPS, mem_caps=MEM_CAPS, seed=SEED, max_tasks=NUM_TASKS)
    obs, _ = env.reset()
    total_latency = 0.0
    done = False
    # always pick node 0 (i.e. FCFS). Replace logic here if you want true FCFS.
    while not done:
        action = 0
        obs, reward, done, truncated, _ = env.step(action)
        total_latency += -reward
    return total_latency

def main():
    rl_lat = run_rl()
    rr_lat = run_rr()
    fcfs_lat = run_fcfs()
    print(f"Over {NUM_TASKS} tasks:")
    print(f"  PPO latency:   {rl_lat:.3f}   (avg {rl_lat/NUM_TASKS:.3f}/task)")
    print(f"  RoundRobin:    {rr_lat:.3f}   (avg {rr_lat/NUM_TASKS:.3f}/task)")
    print(f"  FCFS (node 0): {fcfs_lat:.3f}   (avg {fcfs_lat/NUM_TASKS:.3f}/task)")

if __name__ == "__main__":
    main()
'''