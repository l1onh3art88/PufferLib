import numpy as np
import gymnasium
import pandas as pd
import pufferlib
import os
import yaml
import struct
from pufferlib.ocean.tradesim import binding

class TradeSim(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
            width=576, height=330,log_interval=2000,
            data_path=None, reward_pnl_scale=100.0, reward_illegal_move = 0.0, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf,
            shape=(414,), dtype=np.float64)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.data_path = data_path
        super().__init__(buf)
            
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, width=width, height=height,
            log_interval=log_interval, data_path=data_path, reward_pnl_scale=reward_pnl_scale, 
            reward_illegal_move=reward_illegal_move
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

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ingest_historical_data(path, config_path=None):
    # Load config if provided
    config = None
    if config_path:
        config = load_config(config_path)
    
    df = pd.read_csv(path)
    # Use config values if available, otherwise use defaults
    if config and 'data' in config and 'historical' in config['data']:
        historical_config = config['data']['historical']
        feature_columns = historical_config.get('feature_columns', df.columns.tolist())
        timestamp_column = historical_config.get('timestamp_column', 'timestamp')
        price_column = historical_config.get('price_column', 'price')
        atr_column = historical_config.get('atr_column', 'atr')
        normalize_features = historical_config.get('normalize_features', False)
    else:
        feature_columns = df.columns.tolist()
        timestamp_column = 'timestamp'
        normalize_features = False

    # Ensure timestamp column exists
    if timestamp_column not in df.columns:
        print(f"Timestamp column '{timestamp_column}' not found in data. Using default index.")
    else:
        # Convert timestamp to datetime and sort
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(by=timestamp_column)
    
    # Ensure all required feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    
    print(f"Loaded {len(df)} rows of order book data")
    
    # Select only the feature columns we want
    feature_data = df[feature_columns].copy()
    feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
    #
    prices = df[price_column].to_numpy()
    atrs = df[atr_column].to_numpy()
    timestamps = df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy()  # 'S' is for string/bytes
    regimes = df['regime'].to_numpy()
    # Convert to numpy array
    breakpoint()
    data = feature_data.to_numpy()
    rows, cols = data.shape
    # compute means and stds
    mean = data.mean(axis=0)
    max = data.max(axis=0)
    std = data.std(axis=0)
    denominator = np.where(
        (max == 0) | (max == 1),
        1.0,  # Use 1.0 for max=0 or max=1
        max * 1.25  # Use max*1.25 for all other values
    )
    normalized_data = data.copy()
    normalized_data = data / denominator[None, :]  # Broadcast denominator across rows
    
    # Check if resources/tradesim directory exists, create if not
    os.makedirs("resources/tradesim", exist_ok=True)
    
    # Verify path is writable
    data_path = "resources/tradesim/data.bin"
    try:
        with open(data_path, "wb") as f:
            pass
    except IOError as e:
        raise IOError(f"Unable to write to {data_path}: {e}")
    with open(data_path, "wb") as f:
        # Write rows and cols as int32 (4 bytes each)
        f.write(struct.pack("ii", rows, cols))
        # Write means 
        for i in range(len(mean)):
            f.write(struct.pack("d", mean[i]))
        # Write stds
        for i in range(len(std)):
            f.write(struct.pack("d", std[i]))
        # Write prices
        for i in range(len(prices)):
            f.write(struct.pack("d", prices[i]))
        # Write atrs
        for i in range(len(atrs)):
            f.write(struct.pack("d", atrs[i]))
        # Write timestamps (as char*)
        for timestamp in timestamps:
            timestamp_bytes = timestamp.encode('utf-8')
            # Write the fixed-length timestamp bytes (always 19 bytes)
            f.write(timestamp_bytes)
        # Write regimes
        for regime in regimes:
            f.write(struct.pack("i", regime))
        # Write data
        for i in range(len(normalized_data)):
            for j in range(len(normalized_data[i])):
                f.write(struct.pack("d", normalized_data[i][j]))
        # Write Config Settings
        sim_config = config['simulation']
        f.write(struct.pack("d", sim_config['initial_capital']))
        f.write(struct.pack("d", sim_config['position_size_fixed_dollar']))
        f.write(struct.pack("d", sim_config['pt_atr_mult']))
        f.write(struct.pack("d", sim_config['sl_atr_mult']))
        f.write(struct.pack("i", sim_config['warmup_steps']))
        f.write(struct.pack("d", sim_config['slippage_factor']))
        f.write(struct.pack("d", sim_config['transaction_fee_pct']))
        f.write(struct.pack("i", sim_config['max_steps_per_episode']))
        reward_type = config['rl']['reward_type']
        if(reward_type == 'simple_pnl'):
            reward_type = 0
        elif(reward_type == 'sortino'):
            reward_type = 1
        elif(reward_type == 'sharpe'):
            reward_type = 2
        f.write(struct.pack("i", reward_type))

        
        # f.write()
    return df

if __name__ == '__main__':
    # Example usage with config
    ingest_historical_data(
        "resources/tradesim/train_test_1.csv",
        config_path="resources/tradesim/experiment_config_3.yaml"
    )
