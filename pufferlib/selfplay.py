"""Selfplay-pool training: a fraction of envs play primary vs a frozen historical
snapshot, the rest are pure selfplay. Used by `_train` in pufferl.py — gated on
`selfplay.enabled` (config section).

Pool grows on two triggers:
  - snapshot_interval: every N global steps, save primary weights as a new
    pool entry regardless of winrate. Provides a steady cadence.
  - winrate-driven swap: when primary beats the current opponent at >=
    swap_winrate over >= min_games, also save primary as a pool entry, then
    swap to a new opponent. Marks progress checkpoints in the curriculum.

Swap (without a snapshot) also fires when opp_timeout_steps have elapsed
since the current opponent was finalized. Timeout prevents stalemates from
pinning the curriculum to a single opponent indefinitely.

Pool storage is disk-only (paths held in memory; weights only on GPU when
loaded as the frozen bank). Stride-eviction preserves temporal coverage when
the pool exceeds its cap.
"""
import os

import numpy as np

from pufferlib import _C


def sample_opponent(pool, rng):
    candidates = pool if len(pool) < 6 else pool[:-5]
    weights = np.array([(i + 1) ** 2 for i in range(len(candidates))], dtype=np.float64)
    weights /= weights.sum()
    idx = int(rng.choice(len(candidates), p=weights))
    return candidates[idx]


def update_elo(primary_elo, opp_elo, score_rate, k):
    expected = 1.0 / (1.0 + 10.0 ** ((opp_elo - primary_elo) / 400.0))
    delta = k * (score_rate - expected)
    return primary_elo + delta, opp_elo - delta


def evict(pool, max_size):
    '''Drop every other entry from the oldest half once the pool exceeds max_size.
    Newest half is preserved intact.'''
    if len(pool) <= max_size:
        return pool
    half = len(pool) // 2
    return pool[:half:2] + pool[half:]


def build_perm_tags(num_buffers, agents_per_buffer, agents_per_env, frozen_size, num_envs):
    '''Build env-slot -> rollout-row routing and per-env historical tag.

    Generalizes to any NvN competitive env. Each env has agents_per_env slots
    split into two equal teams of size team_size = agents_per_env // 2:
    slots [0, team_size) are "team A" (always primary); slots [team_size,
    agents_per_env) are "team B" (frozen on historical envs, primary on selfplay).

    Per-buffer physical-row layout (apb = agents_per_buffer):
        [0, apb-2*frozen_size)         primary — selfplay envs (all slots)
        [apb-2*frozen_size, apb-frozen_size) primary — historical envs' team A
        [apb-frozen_size, apb)         frozen  — historical envs' team B

    Env order within a buffer: selfplay envs first (tag=0), then historical
    envs (tag=1). frozen_size counts agent slots (= hist_envs * team_size).
    Primary region [0, apb-frozen_size) matches bank_layout[1] regardless of
    team_size or env order. Returns (perm, tags, num_historical_envs_total).'''
    team_size = agents_per_env // 2
    envs_per_buffer = agents_per_buffer // agents_per_env
    hist_envs_per_buffer = frozen_size // team_size
    selfplay_envs = envs_per_buffer - hist_envs_per_buffer
    perm = np.empty(num_buffers * agents_per_buffer, dtype=np.int32)
    tags = np.zeros(num_envs, dtype=np.int32)
    env_idx = 0
    for b in range(num_buffers):
        buf_start          = b * agents_per_buffer
        hist_primary_start = buf_start + agents_per_buffer - 2 * frozen_size
        frozen_start       = buf_start + agents_per_buffer - frozen_size
        for e in range(envs_per_buffer):
            slot_base = buf_start + e * agents_per_env
            if e < selfplay_envs:
                for s in range(agents_per_env):
                    perm[slot_base + s] = slot_base + s
                tags[env_idx] = 0
            else:
                h = e - selfplay_envs
                for s in range(team_size):
                    perm[slot_base + s] = hist_primary_start + h * team_size + s
                    perm[slot_base + team_size + s] = frozen_start + h * team_size + s
                tags[env_idx] = 1
            env_idx += 1
    return perm, tags, hist_envs_per_buffer * num_buffers


def setup(pufferl, backend, args, run_id):
    '''Wire up agent_perm/tags and bootstrap the frozen bank with the current
    weights so historical envs have an opponent from rollout 1. Returns a
    pool_state dict (or None if disabled).'''
    sp = args.get('selfplay', {})
    if not sp.get('enabled', 0):
        return None
    if backend is not _C:
        raise RuntimeError('selfplay_pool requires the native CUDA backend')

    total_agents = int(args['vec']['total_agents'])
    num_buffers = int(args['vec']['num_buffers'])
    if total_agents % num_buffers != 0:
        raise RuntimeError(f'total_agents ({total_agents}) must be divisible by '
                           f'num_buffers ({num_buffers})')
    agents_per_buffer = total_agents // num_buffers

    num_envs = backend.num_envs(pufferl)
    agents_per_env = total_agents // num_envs
    if agents_per_env % 2 != 0:
        raise RuntimeError(f'agents_per_env ({agents_per_env}) must be even (two equal teams)')
    if agents_per_buffer % agents_per_env != 0:
        raise RuntimeError(f'agents_per_buffer ({agents_per_buffer}) must be divisible by '
                           f'agents_per_env ({agents_per_env})')
    team_size = agents_per_env // 2

    frozen_size = int(agents_per_buffer * float(args['vec']['frozen_bank_pct']))
    # Align to whole teams so each historical env contributes exactly team_size frozen slots.
    frozen_size -= frozen_size % team_size
    if frozen_size <= 0:
        raise RuntimeError('selfplay.enabled but frozen_bank_pct rounds to 0 slots '
                           f'after team-size ({team_size}) alignment')
    if frozen_size >= agents_per_buffer // 2:
        raise RuntimeError(f'frozen_size {frozen_size} >= apb/2 {agents_per_buffer//2}')

    perm, tags, num_hist_envs = build_perm_tags(
        num_buffers, agents_per_buffer, agents_per_env, frozen_size, num_envs)
    backend.set_agent_perm(pufferl, perm)
    backend.set_env_tags(pufferl, tags)

    pool_dir = os.path.join(args['checkpoint_dir'], args['env_name'], run_id, 'pool')
    os.makedirs(pool_dir, exist_ok=True)
    bootstrap_path = os.path.join(pool_dir, f'{pufferl.global_step:016d}.bin')
    backend.save_weights(pufferl, bootstrap_path)
    backend.load_frozen_bank(pufferl, 0, bootstrap_path)

    elo_init = float(sp.get('elo_init', 0.0))
    elo_k    = float(sp.get('elo_k',    16.0))
    rng = np.random.default_rng(int(sp.get('seed', 0)))

    return {
        'pool_dir': pool_dir,
        'pool': [{'path': bootstrap_path, 'elo': elo_init}],
        'rng': rng,
        'max_size': int(sp['max_size']),
        'min_games': int(sp['min_games']),
        'swap_winrate': float(sp['swap_winrate']),
        'snapshot_interval': int(sp.get('snapshot_interval', 1_000_000_000)),
        'opp_timeout_steps': int(sp.get('opp_timeout_steps', 500_000_000)),
        'num_hist_envs': num_hist_envs,
        'hist_score': 0.0,
        'hist_n': 0.0,
        'cur_opp_path': bootstrap_path,
        'cur_opp_elo': elo_init,
        'pending_opp_path': None,
        'pending_opp_elo': None,
        'epoch_armed': 0,
        'primary_elo': elo_init,
        'elo_k': elo_k,
        'last_winrate_at_swap': 0.0,
        'last_epochs_to_align': 0,
        'last_snapshot_step': int(pufferl.global_step),
        'opp_started_step': int(pufferl.global_step),
    }


def step(pufferl, backend, pool_state, flat_logs, epoch):
    if pool_state is None:
        return

    # Recover raw counts: aggregator divides Log fields by total n_window.
    n_window = float(flat_logs.get('env/n', 0.0))
    hist_score_window = float(flat_logs.get('env/hist_score', 0.0)) * n_window
    hist_n_window     = float(flat_logs.get('env/hist_n',     0.0)) * n_window

    if hist_n_window > 0.0:
        pool_state['hist_score'] += hist_score_window
        pool_state['hist_n']     += hist_n_window
        score_rate = hist_score_window / hist_n_window
        new_p, new_o = update_elo(pool_state['primary_elo'],
            pool_state['cur_opp_elo'], score_rate, pool_state['elo_k'])
        pool_state['primary_elo'] = new_p
        pool_state['cur_opp_elo'] = new_o
        for entry in pool_state['pool']:
            if entry['path'] == pool_state['cur_opp_path']:
                entry['elo'] = new_o
                break

    winrate = (pool_state['hist_score'] / pool_state['hist_n']
                   if pool_state['hist_n'] > 0 else None)

    # Snapshot cadence is independent of swap. Anchored to setup-time global_step
    # so the bootstrap entry counts as snapshot 0 and the interval starts there.
    # Set snapshot_interval to 0 to disable interval-based snapshotting.
    if (pool_state['snapshot_interval'] > 0
            and pufferl.global_step - pool_state['last_snapshot_step']
                >= pool_state['snapshot_interval']):
        snap_path = os.path.join(pool_state['pool_dir'],
            f'{pufferl.global_step:016d}.bin')
        backend.save_weights(pufferl, snap_path)
        pool_state['pool'].append({'path': snap_path, 'elo': pool_state['primary_elo']})
        pool_state['pool'] = evict(pool_state['pool'], pool_state['max_size'])
        pool_state['last_snapshot_step'] = int(pufferl.global_step)

    # Swap trigger: winrate breach OR opp_timeout_steps elapsed since the
    # current opponent was finalized. Timeout prevents permanent stalemate on
    # a single opponent. Set opp_timeout_steps to 0 to disable the timeout.
    winrate_met = (winrate is not None
        and pool_state['hist_n'] >= pool_state['min_games']
        and winrate >= pool_state['swap_winrate'])
    timed_out = (pool_state['opp_timeout_steps'] > 0
        and pufferl.global_step - pool_state['opp_started_step']
            >= pool_state['opp_timeout_steps'])

    if pool_state['pending_opp_path'] is not None:
        if backend.count_aligned(pufferl, 1, 0) >= pool_state['num_hist_envs']:
            backend.load_frozen_bank(pufferl, 0, pool_state['pending_opp_path'])
            backend.count_aligned(pufferl, 1, 1)
            pool_state['cur_opp_path'] = pool_state['pending_opp_path']
            pool_state['cur_opp_elo'] = pool_state['pending_opp_elo']
            pool_state['pending_opp_path'] = None
            pool_state['pending_opp_elo'] = None
            pool_state['hist_score'] = 0.0
            pool_state['hist_n'] = 0.0
            pool_state['opp_started_step'] = int(pufferl.global_step)
            pool_state['last_epochs_to_align'] = epoch - pool_state['epoch_armed']
    elif winrate_met or timed_out:
        # Beating the current opponent past swap_winrate adds primary to the
        # pool only while the pool is still small (<  10).
        # After that, only the interval cadence grows the pool — prevents
        # late-training instant-solve cycles from bloating it with duplicates.
        # Timeout-driven swaps never snapshot (stalemate, not progress).
        if winrate_met and len(pool_state['pool']) < 10:
            snap_path = os.path.join(pool_state['pool_dir'],
                f'{pufferl.global_step:016d}.bin')
            backend.save_weights(pufferl, snap_path)
            pool_state['pool'].append({'path': snap_path, 'elo': pool_state['primary_elo']})
            pool_state['pool'] = evict(pool_state['pool'], pool_state['max_size'])
            pool_state['last_snapshot_step'] = int(pufferl.global_step)
        opp_entry = sample_opponent(pool_state['pool'], pool_state['rng'])
        pool_state['pending_opp_path'] = opp_entry['path']
        pool_state['pending_opp_elo'] = opp_entry['elo']
        pool_state['epoch_armed'] = epoch
        pool_state['last_winrate_at_swap'] = winrate if winrate is not None else 0.0

    # Emit at end so dashboard sees the latest values from any updates above.
    # env/* prefix surfaces in dashboard; only learning policy's Elo is reported.
    flat_logs['pool/size']            = len(pool_state['pool'])
    flat_logs['env/elo']              = pool_state['primary_elo']
    flat_logs['pool/winrate_at_swap'] = pool_state['last_winrate_at_swap']
    flat_logs['pool/epochs_to_align'] = pool_state['last_epochs_to_align']
    if winrate is not None:
        flat_logs['pool/winrate']           = winrate
        flat_logs['env/historical_winrate'] = winrate
