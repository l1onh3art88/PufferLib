import numpy as np
import gymnasium
import pandas as pd
import pufferlib
import os
from pufferlib.ocean.tradesim import binding

class TradeSim(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
            width=576, height=330,log_interval=128,
            buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf,
            shape=(414), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0
        self.single_action_space = gymnasium.spaces.Discrete(4)
            
        super().__init__(buf)
            
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, width=width, height=height,
            log_interval=log_interval
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
            
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = TradeSim(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 4, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

def ingest_historical_data(path):
    df = pd.read_csv(path)
    feature_columns = df.columns.tolist()
    # Ensure timestamp column exists
    # if 'timestamp' not in df.columns:
    #     print(f"Timestamp column '{timestamp_column}' not found in data. Using default index.")
    # else:
    # Convert timestamp to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    
    # Ensure all required feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    
    print(f"Loaded {len(df)} rows of order book data")
    data = df.to_numpy().astype(np.float32)
    rows, cols = data.shape
    with open("data_metadata.txt", "w") as f:
        f.write(f"{rows} {cols}")
    
    # Check if resources/tradesim directory exists, create if not
    os.makedirs("resources/tradesim", exist_ok=True)
    
    # Verify path is writable
    data_path = "resources/tradesim/data.bin"
    try:
        with open(data_path, "wb") as f:
            pass
    except IOError as e:
        raise IOError(f"Unable to write to {data_path}: {e}")

    # Save data as raw binary
    data.tofile(data_path)
    return df

if __name__ == '__main__':
    test_performance()
