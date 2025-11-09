import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F


def make_dummy_data(*shape, seed=42):
    np.random.seed(seed)
    ary = np.random.rand(*shape).astype(np.float32) - 0.5
    return np.ascontiguousarray(ary)


def assert_near(a, b, atol=1e-4):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    diff = np.abs(a - b).max()
    assert diff < atol, f"max diff = {diff:.2e} >= {atol}"
    print(f"max diff = {diff:.2e}  ->  PASSED")

# === 1. Simple MLP ===
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=64, h1=128, h2=64, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPLSTMPolicy(nn.Module):
    def __init__(self, in_dim=64, h1=128, h2=64, lstm_dim=32, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, lstm_dim)
        self.lstm = nn.LSTMCell(lstm_dim, lstm_dim)
        self.head = nn.Linear(lstm_dim, out_dim)

    def forward(self, x, hx, cx):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        hx, cx = self.lstm(x, (hx, cx))
        return self.head(hx), (hx, cx)

# === 2. Create policies ===
def create_policies(num_policies=4, seed=0):
    torch.manual_seed(seed)
    policies = [MLPLSTMPolicy() for _ in range(num_policies)]
    with torch.no_grad():
        for i, p in enumerate(policies):
            for param in p.parameters():
                param.add_(i * 0.1)
    return policies

# === 3. Build weight stacks ===
def build_weight_stacks(policies):
    W = {}
    b = {}
    for name in ['fc1', 'fc2', 'fc3', 'head']:
        W[name] = torch.stack([p.state_dict()[f'{name}.weight'] for p in policies])
        b[name] = torch.stack([p.state_dict()[f'{name}.bias']   for p in policies])

    lstm_ih = torch.stack([p.state_dict()['lstm.weight_ih'] for p in policies])
    lstm_hh = torch.stack([p.state_dict()['lstm.weight_hh'] for p in policies])
    lstm_b_ih = torch.stack([p.state_dict()['lstm.bias_ih'] for p in policies])
    lstm_b_hh = torch.stack([p.state_dict()['lstm.bias_hh'] for p in policies])
    return {'W': W, 'b': b, 
            'lstm_ih': lstm_ih, 'lstm_hh': lstm_hh, 
            'lstm_b_ih': lstm_b_ih, 'lstm_b_hh': lstm_b_hh}

# === 4. Optimized Looped Forward (Sorted + Grouped) ===
def optimized_loop_forward(obs_sorted, version_ids,h_sorted, c_sorted, policies):
    out = torch.empty_like(obs_sorted[:, :4])  # pre-allocate
    new_h = torch.empty_like(h_sorted)
    new_c = torch.empty_like(c_sorted)
    start = 0
    for v in torch.unique(version_ids):
        mask = (version_ids == v)
        end = start + mask.sum().item()
        sub_obs = obs_sorted[start:end]
        sub_h = h_sorted[start:end]
        sub_c = c_sorted[start:end]
        with torch.no_grad():
            #out[start:end] = policies[v](sub_obs)
            act, (nh, nc) = policies[v](sub_obs, sub_h, sub_c)
        out[start:end] = act
        new_h[start:end] = nh
        new_c[start:end] = nc
        start = end
    return out, new_h, new_c

def stacked_forward_mlp(obs, version_ids, stacks):
    idx = torch.as_tensor(version_ids, device=obs.device, dtype=torch.long)


    x = obs

    for name in ['fc1', 'fc2']:
        W = stacks['W'][name][idx]
        b = stacks['b'][name][idx]
        x = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b
        #x = torch.baddbmm(b.unsqueeze(1), x.unsqueeze(1), W.transpose(-2, -1)).squeeze(1)

        x = F.relu(x)

    # fc3
    W = stacks['W']['fc3'][idx]
    b = stacks['b']['fc3'][idx]
    out = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b
    #out = torch.baddbmm(b.unsqueeze(1), x.unsqueeze(1), W.transpose(-2, -1)).squeeze(1)


    return out

def stacked_forward_mlp_lstm(obs, version_ids, h, c, stacks):
    idx = torch.as_tensor(version_ids, device=obs.device, dtype=torch.long)
    x = obs

    # MLP
    for name in ('fc1', 'fc2', 'fc3'):
        W = stacks['W'][name][idx]
        b = stacks['b'][name][idx]
        x = torch.bmm(x.unsqueeze(1), W.transpose(-2, -1)).squeeze(1) + b
        if name != 'fc3':
            x = F.relu(x)

    # LSTM: 4 linear ops
    W_ih = stacks['lstm_ih'][idx]      # [B, 4*H, D]
    W_hh = stacks['lstm_hh'][idx]      # [B, 4*H, H]
    b_ih = stacks['lstm_b_ih'][idx]    # [B, 4*H]
    b_hh = stacks['lstm_b_hh'][idx]    # [B, 4*H]

    gates = torch.bmm(x.unsqueeze(1), W_ih.transpose(-2, -1)).squeeze(1) + \
            torch.bmm(h.unsqueeze(1), W_hh.transpose(-2, -1)).squeeze(1) + \
            b_ih + b_hh

    i, f, g, o = gates.chunk(4, dim=-1)
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    new_c = f * c + i * g
    new_h = o * torch.tanh(new_c)

    # Head
    W = stacks['W']['head'][idx]
    b = stacks['b']['head'][idx]
    out = torch.bmm(new_h.unsqueeze(1), W.transpose(-2, -1)).squeeze(1) + b

    return out, new_h, new_c

# === 6. Benchmark ===
def benchmark(num_iters=1000, batch_size=512, num_versions=4):
    policies = create_policies(num_policies=num_versions)
    stacks = build_weight_stacks(policies)

    # Generate data
    obs = torch.randn(batch_size, 64, device='cuda')
    h = torch.randn(batch_size, 32, device='cuda')
    c = torch.randn(batch_size, 32, device='cuda')
    versions = np.random.randint(0, num_versions, size=batch_size)

    # === SORT for fair optimized loop ===
    sort_idx = np.argsort(versions)
    obs_sorted = obs.cpu()[sort_idx]
    h_sorted = h.cpu()[sort_idx]
    c_sorted = c.cpu()[sort_idx]
    versions_sorted = versions[sort_idx]

    gpu_stacks = {}
    for k, v in stacks.items():
        if isinstance(v, dict):
            gpu_stacks[k] = {kk: vv.to('cuda') for kk, vv in v.items()}
        else:
            gpu_stacks[k] = v.to('cuda')


    # Warmup
    for _ in range(10):
        optimized_loop_forward(obs_sorted.cpu(), torch.from_numpy(versions_sorted), h_sorted, c_sorted, [p.cpu() for p in policies])
        stacked_forward_mlp_lstm(obs.to('cuda'), torch.from_numpy(versions), h, c, gpu_stacks)

    # === TIME OPTIMIZED LOOP ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        optimized_loop_forward(obs_sorted.cpu(), torch.from_numpy(versions_sorted), h_sorted, c_sorted, [p.cpu() for p in policies])
    torch.cuda.synchronize()
    opt_loop_time = time.time() - t0

    # === TIME STACKED ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        stacked_forward_mlp_lstm(obs, torch.from_numpy(versions), h, c, gpu_stacks)
    torch.cuda.synchronize()
    stack_time = time.time() - t0

    print(f"\n=== FAIR BENCHMARK (iters={num_iters}, B={batch_size}, V={num_versions}) ===")
    print(f"Old Method: {opt_loop_time:.3f}s  ->  {opt_loop_time/num_iters*1000:.2f} ms/step")
    print(f"New Method: {stack_time:.3f}s  ->  {stack_time/num_iters*1000:.2f} ms/step")
    print(f"Speedup        : {opt_loop_time/stack_time:.1f}x")

# === 7. MAIN ===
if __name__ == "__main__":
    # --- CORRECTNESS: GPU vs GPU ---
    print("=== CORRECTNESS (GPU vs GPU) ===")
    policies = create_policies(4)
    policies_gpu = [p.to('cuda') for p in policies] 

    stacks = build_weight_stacks(policies)
    
    gpu_stacks = {}
    for k, v in stacks.items():
        if isinstance(v, dict):
            gpu_stacks[k] = {kk: vv.to('cuda') for kk, vv in v.items()}
        else:
            gpu_stacks[k] = v.to('cuda')

    B = 256
    obs = torch.randn(B, 64, device='cuda')
    h = torch.randn(B, 32, device='cuda')
    c = torch.randn(B, 32, device='cuda')
    versions = np.random.randint(0, 4, size=B)

    sort_idx = np.argsort(versions)
    obs_sorted = obs[sort_idx]
    h_sorted = h[sort_idx]
    c_sorted = c[sort_idx]
    versions_sorted = versions[sort_idx]

    # Loop on GPU
    loop_out, loop_h, loop_c = optimized_loop_forward(
        obs_sorted,
        torch.from_numpy(versions_sorted).to('cuda'),
        h_sorted, c_sorted,
        policies_gpu
    )

    # Stack on GPU
    stack_out, stack_h, stack_c = stacked_forward_mlp_lstm(
        obs,
        torch.from_numpy(versions).to('cuda'),
        h, c,
        gpu_stacks
    )

    # Unsort loop output
    unsort_idx = np.argsort(sort_idx)
    loop_out = loop_out[unsort_idx]
    loop_h = loop_h[unsort_idx]
    loop_c = loop_c[unsort_idx]

    assert_near(loop_out.cpu().numpy(), stack_out.cpu().numpy(), atol=1e-4)
    assert_near(loop_h.cpu().numpy(), stack_h.cpu().numpy(), atol=1e-4)
    assert_near(loop_c.cpu().numpy(), stack_c.cpu().numpy(), atol=1e-4)
    benchmark(num_iters=1000, batch_size=1024, num_versions=4)


