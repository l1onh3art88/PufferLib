"""Selfplay-pool training: a fraction of envs play primary vs a frozen historical
snapshot, the rest are pure selfplay. Used by `_train` in pufferl.py — gated on
`selfplay.enabled` (config section).

Pool growth and opponent swaps are decoupled:
  - snapshot_interval: every N global steps, each rank saves primary weights
    into its own pool dir (post-allreduce weights match; no cross-rank file sync).
  - opp_timeout_steps: every N global steps per bank, ranks sample new opponents
    on a fixed cadence and load them after all historical envs reach an episode
    boundary. 0 disables fixed-interval swapping.

Opponent sampling is configurable:
  - sample = uniform: equal probability over pool entries
  - sample = pfsp: prioritized fictitious self-play — weight opponents by how
    hard they are for the current learner (from per-checkpoint matchup stats).
    Snapshot/swap timing is still fixed; winrate only affects *who* is sampled.

Each rank owns `.../pool/rank_{rank}/`. Sampling is f(seed, rank, draw_slot)
with draw_slot = agent_step // opp_timeout_steps. No shared JSON / publish.
"""
import glob
import os
import shutil

import numpy as np

from pufferlib import _C


def make_pool_entry(path, learner_score=0.0, learner_n=0.0):
    '''Pool entry: checkpoint path + learner matchup stats vs that checkpoint.

    learner_score / learner_n ≈ primary winrate while that opp was loaded
    (same units as env/hist_score: weighted episode outcomes).'''
    return {
        'path': path,
        'learner_score': float(learner_score),
        'learner_n': float(learner_n),
    }


def _file_nbytes(path):
    try:
        return os.path.getsize(path) if path and os.path.isfile(path) else -1
    except OSError:
        return -1


def filter_pool_by_nbytes(pool, expected_nbytes, label='pool'):
    '''Drop checkpoints whose weight file size != frozen bank / primary size.

    Avoids pufferl_load_frozen_bank size-mismatch spam when opponent_pool (or a
    shared pool) mixes arches from different Protein trials.'''
    if expected_nbytes is None or expected_nbytes <= 0:
        return list(pool or [])
    kept, dropped = [], 0
    for e in pool or []:
        path = e.get('path') if isinstance(e, dict) else e
        if _file_nbytes(path) == int(expected_nbytes):
            kept.append(e if isinstance(e, dict) else make_pool_entry(path))
        else:
            dropped += 1
    if dropped:
        print(
            f'[selfplay] dropped {dropped}/{dropped + len(kept)} {label} '
            f'checkpoints with weight size != {int(expected_nbytes)} bytes '
            f'(frozen-bank arch mismatch)'
        )
    return kept


def learner_winrate(entry):
    n = float(entry.get('learner_n', 0.0) or 0.0)
    if n <= 0.0:
        return None
    return float(entry.get('learner_score', 0.0) or 0.0) / n


def pfsp_weights(win_rates, weighting='linear', power=1.0, epsilon=0.05, explore=1.0):
    '''PFSP sampling weights from learner winrates vs each opponent.

    win_rates[i] is primary's WR against pool[i], or None if never played.
    Weightings (AlphaStar-style family):
      linear:   max(eps, 1 - wr)^power   — prefer hard opponents
      squared:  max(eps, 1 - wr)^2
      variance: max(eps, 1 - |2wr-1|)    — prefer near-50% matchups
    Unseen opponents get `explore` weight (default 1.0) so the pool is probed.
    '''
    wr = np.asarray(win_rates, dtype=np.float64)
    n = len(wr)
    if n == 0:
        return wr
    # None / nan → unseen
    unseen = np.array([w is None or (isinstance(w, float) and np.isnan(w))
                       for w in win_rates], dtype=bool)
    # For seen, clip to [0,1] and quantize so tiny CUDA/FP noise in matchup
    # stats cannot flip PFSP multinomial draws across otherwise-identical runs.
    seen_wr = np.zeros(n, dtype=np.float64)
    for i, w in enumerate(win_rates):
        if not unseen[i]:
            seen_wr[i] = round(float(np.clip(w, 0.0, 1.0)), 4)

    weighting = str(weighting).lower()
    eps = float(epsilon)
    p = float(power)
    if weighting in ('linear', 'hardest'):
        w = np.maximum(eps, 1.0 - seen_wr) ** p
    elif weighting in ('squared', 'square'):
        w = np.maximum(eps, 1.0 - seen_wr) ** 2.0
    elif weighting in ('variance', 'focus', 'var'):
        w = np.maximum(eps, 1.0 - np.abs(2.0 * seen_wr - 1.0))
    else:
        raise ValueError(f'unknown pfsp weighting: {weighting!r} '
                         f'(expected linear|squared|variance)')

    w = np.where(unseen, float(explore), w)
    total = float(w.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(n, 1.0 / n)
    return w / total


def sample_opponent(pool, rng, sample='uniform', pfsp_weighting='linear',
                    pfsp_power=1.0, pfsp_epsilon=0.05, pfsp_explore=1.0):
    if not pool:
        raise RuntimeError('selfplay opponent pool is empty')
    sample = str(sample or 'uniform').lower()
    if sample in ('uniform', 'uni', 'u'):
        return pool[int(rng.integers(len(pool)))]
    if sample in ('pfsp', 'prioritized', 'priority'):
        win_rates = [learner_winrate(e) for e in pool]
        weights = pfsp_weights(
            win_rates, weighting=pfsp_weighting, power=pfsp_power,
            epsilon=pfsp_epsilon, explore=pfsp_explore)
        idx = int(rng.choice(len(pool), p=weights))
        return pool[idx]
    raise ValueError(f'unknown selfplay.sample={sample!r} (expected uniform|pfsp)')


def sample_opponent_reproducible(pool, seed, rank, draw_slot, **sample_kwargs):
    '''Per-rank different opponents, bit-stable across runs.

    draw_slot is step-locked (agent_step // opp_timeout_steps), not a counter.
    '''
    rng = np.random.default_rng(
        int(seed) + 1_000_003 * int(rank) + 9_911 * int(draw_slot))
    return sample_opponent(pool, rng, **sample_kwargs)


def _find_pool_entry(pool, path):
    for e in pool:
        if e.get('path') == path:
            return e
    return None


def _record_matchup(pool, path, score, n):
    '''Accumulate learner score/n against the checkpoint at path.'''
    if n <= 0.0 or not path:
        return
    e = _find_pool_entry(pool, path)
    if e is None:
        return
    e['learner_score'] = float(e.get('learner_score', 0.0) or 0.0) + float(score)
    e['learner_n'] = float(e.get('learner_n', 0.0) or 0.0) + float(n)


def resolve_opponent_pool(spec):
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        candidates = []
        for item in spec:
            candidates.extend(resolve_opponent_pool(item))
        return sorted(dict.fromkeys(candidates))

    raw_spec = os.path.expanduser(str(spec))

    def _expand(path_spec):
        if os.path.isdir(path_spec):
            return glob.glob(os.path.join(path_spec, '*.bin'))
        candidates = glob.glob(path_spec)
        if not candidates and os.path.isfile(path_spec):
            candidates = [path_spec]
        return candidates

    candidates = _expand(raw_spec)
    if not candidates and not os.path.isabs(raw_spec):
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        candidates = _expand(os.path.join(repo_root, raw_spec))

    return sorted(os.path.abspath(path) for path in candidates
                  if path.endswith('.bin') and os.path.isfile(path))


def evict(pool, max_size):
    '''Drop every other entry from the oldest half once the pool exceeds max_size.
    Newest half is preserved intact.'''
    if len(pool) <= max_size:
        return pool
    half = len(pool) // 2
    return pool[:half:2] + pool[half:]


def mix_env_mode(env_idx, mix_bot_pct, mix_hist_pct):
    '''Mirror ocean/robocode/binding.c mix_enabled composition (env index hash).

    Returns 'bot' | 'hist' | 'sp'.'''
    r = int(env_idx) % 100
    bot_pct = max(0, min(100, int(mix_bot_pct)))
    hist_pct = max(0, min(100 - bot_pct, int(mix_hist_pct)))
    if r < bot_pct:
        return 'bot'
    if r < bot_pct + hist_pct:
        return 'hist'
    return 'sp'


def simulate_mix_layout(total_agents, mix_bot_pct, mix_hist_pct):
    '''Replay my_vec_init packing: create envs until agents_created >= total_agents.

    Matches binding.c mix_enabled + vecenv my_vec_init loop.'''
    envs = []
    agents = 0
    idx = 0
    while agents < total_agents:
        mode = mix_env_mode(idx, mix_bot_pct, mix_hist_pct)
        n_ag = 1 if mode == 'bot' else 2
        envs.append({'idx': idx, 'mode': mode, 'num_agents': n_ag})
        agents += n_ag
        idx += 1
    return envs, agents


def build_perm_tags(num_buffers, agents_per_buffer, agents_per_env, frozen_sizes, num_envs):
    '''Build env-slot -> rollout-row routing and per-env bank tag.

    Multi-bank generalization. `frozen_sizes` is a list of per-bank agent counts
    (per buffer). With one bank this reduces to the legacy single-bank layout.

    Per-buffer physical-row layout (apb = agents_per_buffer, F = sum(frozen_sizes)):
        [0,           apb - 2F)                       primary — selfplay envs (all slots)
        [apb - 2F,    apb - F)                        primary — historical envs' team A
        [apb - F,     apb - F + frozen_sizes[0])      bank 0  — historical envs' team B
        [apb - F + frozen_sizes[0], ... + ...[1])     bank 1  — ... etc.
    F may equal apb//2 (then SP range is empty — 100% hist). F > apb//2 is invalid.

    Env order within a buffer: selfplay envs first (tag=0), then historical
    envs assigned to banks in block order — the first `frozen_sizes[0]/team_size`
    historical envs play bank 0 (tag=1), next block plays bank 1 (tag=2), etc.

    The C-side bank_layout (pufferlib.cu:1798-1806) lays banks out sequentially
    after primary, so our routing matches: bank b's slice is
    [apb - F + sum(frozen_sizes[:b]),  apb - F + sum(frozen_sizes[:b+1])).

    Returns (perm, tags, num_hist_envs_per_bank) — last is a list of per-bank
    historical-env counts across all buffers, used by selfplay.step to know how
    many env alignments to wait for per bank during swaps.'''
    team_size = agents_per_env // 2
    envs_per_buffer = agents_per_buffer // agents_per_env
    num_banks = len(frozen_sizes)
    total_frozen = sum(frozen_sizes)
    hist_envs_per_bank_per_buffer = [fs // team_size for fs in frozen_sizes]
    total_hist_envs_per_buffer = sum(hist_envs_per_bank_per_buffer)
    selfplay_envs = envs_per_buffer - total_hist_envs_per_buffer
    perm = np.empty(num_buffers * agents_per_buffer, dtype=np.int32)
    tags = np.zeros(num_envs, dtype=np.int32)
    env_idx = 0
    for b_buf in range(num_buffers):
        buf_start          = b_buf * agents_per_buffer
        hist_primary_start = buf_start + agents_per_buffer - 2 * total_frozen
        bank_starts = []
        offset = buf_start + agents_per_buffer - total_frozen
        for bank in range(num_banks):
            bank_starts.append(offset)
            offset += frozen_sizes[bank]
        h_within_buffer = 0
        for e in range(envs_per_buffer):
            slot_base = buf_start + e * agents_per_env
            if e < selfplay_envs:
                for s in range(agents_per_env):
                    perm[slot_base + s] = slot_base + s
                tags[env_idx] = 0
            else:
                # Block assignment: walk cumulative bank capacity to find which
                # bank this historical env belongs to.
                bank_idx = 0
                cum = hist_envs_per_bank_per_buffer[0]
                while h_within_buffer >= cum and bank_idx < num_banks - 1:
                    bank_idx += 1
                    cum += hist_envs_per_bank_per_buffer[bank_idx]
                h_in_bank = h_within_buffer - (cum - hist_envs_per_bank_per_buffer[bank_idx])
                team_a_offset = hist_primary_start + h_within_buffer * team_size
                team_b_offset = bank_starts[bank_idx] + h_in_bank * team_size
                for s in range(team_size):
                    perm[slot_base + s] = team_a_offset + s
                    perm[slot_base + team_size + s] = team_b_offset + s
                tags[env_idx] = bank_idx + 1
                h_within_buffer += 1
            env_idx += 1
    num_hist_envs_per_bank = [n * num_buffers for n in hist_envs_per_bank_per_buffer]
    return perm, tags, num_hist_envs_per_bank


def build_perm_tags_mixed(num_buffers, agents_per_buffer, env_specs, frozen_sizes):
    '''Perm/tags for heterogeneous robocode mix (1-agent bot + 2-agent SP/hist).

    env_specs: list of dicts with keys mode ('bot'|'sp'|'hist') and num_agents.
    Order matches env creation order. Physical slots match vecenv packing:
    within each buffer, envs are laid out at buf_start + offset (not a single
    global contiguous pack across buffers). Identity routing by default; hist
    envs' slot 1 is remapped into frozen banks (team_size=1).

    Bot and live-SP envs keep tag=0. Hist envs get tag=bank+1. If there are
    more hist envs than frozen capacity, extras are demoted to live SP (tag=0).
    '''
    num_envs = len(env_specs)
    total_agents = num_buffers * agents_per_buffer
    team_size = 1
    num_banks = max(1, len(frozen_sizes))
    total_frozen = sum(frozen_sizes) if frozen_sizes else 0
    hist_cap_per_buffer = [max(0, fs // team_size) for fs in frozen_sizes] if frozen_sizes else [0]
    hist_cap_total_per_buffer = sum(hist_cap_per_buffer)
    primary_per_buffer = agents_per_buffer - total_frozen
    if primary_per_buffer < 1:
        primary_per_buffer = agents_per_buffer

    # Assign envs to buffers (mirror binding.c / vecenv: pack by agent count
    # against primary_per_buffer, not full agents_per_buffer).
    env_buffer = [0] * num_envs
    buf = 0
    buf_agents = 0
    for i, spec in enumerate(env_specs):
        env_buffer[i] = buf
        buf_agents += int(spec['num_agents'])
        if buf_agents >= primary_per_buffer and buf < num_buffers - 1:
            buf += 1
            buf_agents = 0

    # Slot base within each buffer (buf_start + local offset).
    slot_base = [0] * num_envs
    local_off = [0] * num_buffers
    for i, spec in enumerate(env_specs):
        b = env_buffer[i]
        slot_base[i] = b * agents_per_buffer + local_off[b]
        local_off[b] += int(spec['num_agents'])

    perm = np.arange(total_agents, dtype=np.int32)
    tags = np.zeros(num_envs, dtype=np.int32)

    h_count = [0] * num_buffers
    bank_h_count = [[0] * num_banks for _ in range(num_buffers)]
    num_hist_envs_per_bank = [0] * num_banks

    for i, spec in enumerate(env_specs):
        mode = spec['mode']
        n_ag = int(spec['num_agents'])
        b_buf = env_buffer[i]
        base = slot_base[i]
        for s in range(n_ag):
            perm[base + s] = base + s

        if mode != 'hist' or n_ag < 2 or total_frozen <= 0:
            tags[i] = 0
            continue

        if h_count[b_buf] >= hist_cap_total_per_buffer:
            tags[i] = 0
            continue

        bank_idx = 0
        while (bank_idx < num_banks - 1
               and bank_h_count[b_buf][bank_idx] >= hist_cap_per_buffer[bank_idx]):
            bank_idx += 1
        if bank_h_count[b_buf][bank_idx] >= hist_cap_per_buffer[bank_idx]:
            tags[i] = 0
            continue

        h_in_bank = bank_h_count[b_buf][bank_idx]
        buf_start = b_buf * agents_per_buffer
        bank_offset = buf_start + agents_per_buffer - total_frozen
        for b in range(bank_idx):
            bank_offset += frozen_sizes[b]
        team_b_phys = bank_offset + h_in_bank * team_size

        perm[base + 1] = team_b_phys
        tags[i] = bank_idx + 1

        bank_h_count[b_buf][bank_idx] += 1
        h_count[b_buf] += 1
        num_hist_envs_per_bank[bank_idx] += 1

    return perm, tags, num_hist_envs_per_bank


def make_bank_state(path, current_agent_step, num_hist_envs):
    return {
        'cur_opp_path': path,
        'hist_score': 0.0,
        'hist_n': 0.0,
        'pending_opp_path': None,
        'epoch_armed': 0,
        'opp_started_step': current_agent_step,
        'num_hist_envs': num_hist_envs,
        'last_winrate_at_swap': 0.0,
        'last_epochs_to_align': 0,
    }


def _external_pool_entries(sp):
    spec = sp.get('opponent_pool', '')
    paths = resolve_opponent_pool(spec)
    if spec and not paths:
        raise RuntimeError(f'selfplay.opponent_pool resolved no .bin files: {spec}')
    return [make_pool_entry(path) for path in paths]


def setup(pufferl, backend, args, run_id, artifact_owner=True):
    '''Wire up agent_perm/tags and bootstrap each rank's local frozen pool.

    artifact_owner is ignored for pool I/O (kept so callers need not change).
    Each rank writes `.../pool/rank_{rank}/` from identical post-allreduce weights.
    '''
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
    env_cfg = args.get('env', {})
    mix_enabled = bool(int(float(env_cfg.get('mix_enabled', 0) or 0)))

    num_banks = int(args['vec'].get('num_frozen_banks', 1))
    if num_banks <= 0:
        raise RuntimeError('selfplay.enabled requires num_frozen_banks >= 1')
    if num_banks > 8:
        raise RuntimeError(f'num_frozen_banks {num_banks} exceeds chess.h CHESS_MAX_BANKS=8')

    if mix_enabled:
        # Heterogeneous robocode mix: 1-agent bot envs + 2-agent SP/hist.
        mix_bot_pct = int(float(env_cfg.get('mix_bot_pct', 20) or 0))
        mix_hist_pct = int(float(env_cfg.get('mix_hist_pct', 30) or 0))
        frozen_size = int(agents_per_buffer * float(args['vec'].get('frozen_bank_pct', 0) or 0))
        frozen_size = max(0, frozen_size)
        team_size = 1
        frozen_size -= frozen_size % team_size
        if mix_hist_pct > 0 and frozen_size <= 0:
            raise RuntimeError(
                'mix_hist_pct > 0 requires vec.frozen_bank_pct > 0 and '
                'num_frozen_banks >= 1 so historical envs have frozen opponents')
        # No hist demand → no frozen banks (bot + live SP only).
        if mix_hist_pct <= 0:
            frozen_size = 0
            num_banks = 0
        frozen_sizes = [frozen_size] * max(num_banks, 1) if frozen_size > 0 else [0]
        total_frozen = sum(frozen_sizes) if frozen_size > 0 else 0
        # Layout needs 2F slots (hist team A + frozen team B). F == apb/2 is
        # valid (100% hist, no live SP); only F > apb/2 overflows the buffer.
        if total_frozen > agents_per_buffer // 2 and mix_hist_pct > 0:
            raise RuntimeError(
                f'total_frozen {total_frozen} (> apb/2={agents_per_buffer//2}); '
                f'lower frozen_bank_pct or num_frozen_banks (max frozen_bank_pct=0.5)')

        # Replay C packing into primary capacity only.
        primary_per_buffer = agents_per_buffer - total_frozen
        if primary_per_buffer < 1:
            primary_per_buffer = agents_per_buffer
        primary_cap = primary_per_buffer * num_buffers
        env_specs = []
        agents_created = 0
        idx = 0
        while agents_created < primary_cap:
            remaining = primary_cap - agents_created
            id_for_mode = idx
            if remaining == 1:
                if mix_bot_pct > 0:
                    id_for_mode = 0
                else:
                    break
            mode = mix_env_mode(id_for_mode, mix_bot_pct, mix_hist_pct)
            n_ag = 1 if mode == 'bot' else 2
            if agents_created + n_ag > primary_cap:
                break
            env_specs.append({'idx': idx, 'mode': mode, 'num_agents': n_ag})
            agents_created += n_ag
            idx += 1

        if len(env_specs) != num_envs:
            raise RuntimeError(
                f'mix layout mismatch: python simulated {len(env_specs)} envs, '
                f'backend has {num_envs}. Check mix_* config matches binding.c.')

        perm, tags, num_hist_envs_per_bank = build_perm_tags_mixed(
            num_buffers, agents_per_buffer, env_specs,
            frozen_sizes if frozen_size > 0 else [0])
        backend.set_agent_perm(pufferl, perm)
        backend.set_env_tags(pufferl, tags)
        if frozen_size <= 0:
            num_hist_envs_per_bank = [0]
            num_banks = 0
        n_bot = sum(1 for e in env_specs if e['mode'] == 'bot')
        n_hist = int((tags > 0).sum())
        n_sp = num_envs - n_bot - n_hist
        print(f'[mix] envs={num_envs} bot={n_bot} live_sp≈{n_sp} hist={n_hist} '
              f'(bot_pct={mix_bot_pct} hist_pct={mix_hist_pct})')
    else:
        agents_per_env = total_agents // num_envs
        if agents_per_env % 2 != 0:
            raise RuntimeError(
                f'agents_per_env ({agents_per_env}) must be even (two equal teams)')
        if agents_per_buffer % agents_per_env != 0:
            raise RuntimeError(
                f'agents_per_buffer ({agents_per_buffer}) must be divisible by '
                f'agents_per_env ({agents_per_env})')
        team_size = agents_per_env // 2

        # frozen_bank_pct is per-bank (matches C-side: pufferlib.cu:2069).
        frozen_size = int(agents_per_buffer * float(args['vec']['frozen_bank_pct']))
        frozen_size -= frozen_size % team_size
        if frozen_size <= 0:
            raise RuntimeError(
                'selfplay.enabled but frozen_bank_pct rounds to 0 slots '
                f'after team-size ({team_size}) alignment')
        total_frozen = frozen_size * num_banks
        if total_frozen > agents_per_buffer // 2:
            raise RuntimeError(
                f'total_frozen {total_frozen} (= num_banks {num_banks} '
                f'* per_bank {frozen_size}) > apb/2 {agents_per_buffer//2} '
                f'(max frozen_bank_pct=0.5 with one bank)')

        frozen_sizes = [frozen_size] * num_banks
        perm, tags, num_hist_envs_per_bank = build_perm_tags(
            num_buffers, agents_per_buffer, agents_per_env, frozen_sizes, num_envs)
        backend.set_agent_perm(pufferl, perm)
        backend.set_env_tags(pufferl, tags)

    rank = int(args.get('rank', 0))
    sp_seed = int(sp.get('seed', 0))
    world_size = max(1, int(args.get('world_size', 1)))
    current_agent_step = int(pufferl.global_step) * world_size
    # Per-rank local pool — no cross-rank publish/wait.
    pool_dir = os.path.join(
        args['checkpoint_dir'], args['env_name'], run_id, 'pool', f'rank_{rank}')

    sample = str(sp.get('sample', 'uniform')).lower()
    pfsp_weighting = str(sp.get('pfsp_weighting', 'linear')).lower()
    pfsp_power = float(sp.get('pfsp_power', 1.0))
    pfsp_epsilon = float(sp.get('pfsp_epsilon', 0.05))
    pfsp_explore = float(sp.get('pfsp_explore', 1.0))
    if sample not in ('uniform', 'uni', 'u', 'pfsp', 'prioritized', 'priority'):
        raise RuntimeError(f'selfplay.sample must be uniform|pfsp (got {sample!r})')

    banks_state = []
    external_pool = _external_pool_entries(sp)
    sample_kwargs = dict(
        sample=sample, pfsp_weighting=pfsp_weighting, pfsp_power=pfsp_power,
        pfsp_epsilon=pfsp_epsilon, pfsp_explore=pfsp_explore,
    )
    opp_timeout = int(sp.get('opp_timeout_steps', 500_000_000))
    expected_nbytes = None
    pool = []
    if num_banks > 0:
        os.makedirs(pool_dir, exist_ok=True)
        bootstrap_path = os.path.join(
            pool_dir, f'{pufferl.global_step:016d}.bin')
        backend.save_weights(pufferl, bootstrap_path)
        expected_nbytes = _file_nbytes(bootstrap_path)
        if external_pool:
            pool = filter_pool_by_nbytes(
                external_pool, expected_nbytes, label='opponent_pool')
            if not pool:
                print(
                    f'[selfplay] opponent_pool had 0 size-compatible checkpoints; '
                    f'bootstrapping from primary ({expected_nbytes} bytes)'
                )
                pool = [make_pool_entry(bootstrap_path)]
        else:
            pool = [make_pool_entry(bootstrap_path)]

    draw_slot = (current_agent_step // opp_timeout) if opp_timeout > 0 else 0
    for b in range(num_banks):
        # Opening draw: slot + bank index so banks differ without a counter.
        opp_entry = sample_opponent_reproducible(
            pool, sp_seed, rank, draw_slot + b, **sample_kwargs)
        backend.load_frozen_bank(pufferl, b, opp_entry['path'])
        banks_state.append(make_bank_state(
            opp_entry['path'], current_agent_step, num_hist_envs_per_bank[b]))

    return {
        'pool_dir': pool_dir,
        'pool': pool,
        'seed': sp_seed,
        'rank': rank,
        'max_size': int(sp['max_size']),
        'snapshot_interval': int(sp.get('snapshot_interval', 1_000_000_000)),
        'opp_timeout_steps': opp_timeout,
        'sample': sample,
        'pfsp_weighting': pfsp_weighting,
        'pfsp_power': pfsp_power,
        'pfsp_epsilon': pfsp_epsilon,
        'pfsp_explore': pfsp_explore,
        'num_banks': num_banks,
        'banks': banks_state,
        'world_size': world_size,
        'last_snapshot_step': current_agent_step,
        'expected_nbytes': expected_nbytes,
        'sample_kwargs': sample_kwargs,
    }


def sync(pufferl, backend, pool_state):
    # Local pools — nothing to pull from other ranks.
    return


def cleanup(pool_state):
    '''Drop duplicate per-rank pool dirs after train. Keep rank_0 as the archive.'''
    if not pool_state:
        return
    if int(pool_state.get('rank', 0)) == 0:
        return
    pool_dir = pool_state.get('pool_dir')
    if pool_dir:
        shutil.rmtree(pool_dir, ignore_errors=True)


def step(pufferl, backend, pool_state, flat_logs, epoch):
    if pool_state is None:
        return

    n_window = float(flat_logs.get('env/n', 0.0))
    num_banks = pool_state['num_banks']
    current_agent_step = int(pufferl.global_step) * int(pool_state.get('world_size', 1))
    opp_timeout = int(pool_state['opp_timeout_steps'])

    for b in range(num_banks):
        bank = pool_state['banks'][b]
        hist_score_w = float(flat_logs.get(f'env/hist_score_bank_{b}', 0.0)) * n_window
        hist_n_w = float(flat_logs.get(f'env/hist_n_bank_{b}', 0.0)) * n_window
        if hist_n_w > 0.0:
            bank['hist_score'] += hist_score_w
            bank['hist_n'] += hist_n_w
            _record_matchup(pool_state['pool'], bank.get('cur_opp_path'),
                            hist_score_w, hist_n_w)

    if (pool_state['snapshot_interval'] > 0
            and current_agent_step - pool_state['last_snapshot_step']
                >= pool_state['snapshot_interval']):
        os.makedirs(pool_state['pool_dir'], exist_ok=True)
        snap_path = os.path.join(pool_state['pool_dir'],
            f'{pufferl.global_step:016d}.bin')
        backend.save_weights(pufferl, snap_path)
        pool_state['pool'].append(make_pool_entry(snap_path))
        pool_state['pool'] = evict(pool_state['pool'], pool_state['max_size'])
        pool_state['last_snapshot_step'] = current_agent_step

    sample_kwargs = pool_state.get('sample_kwargs') or dict(
        sample=pool_state.get('sample', 'uniform'),
        pfsp_weighting=pool_state.get('pfsp_weighting', 'linear'),
        pfsp_power=pool_state.get('pfsp_power', 1.0),
        pfsp_epsilon=pool_state.get('pfsp_epsilon', 0.05),
        pfsp_explore=pool_state.get('pfsp_explore', 1.0),
    )
    for b in range(num_banks):
        bank = pool_state['banks'][b]
        winrate = (bank['hist_score'] / bank['hist_n']
                   if bank['hist_n'] > 0 else None)
        timed_out = (opp_timeout > 0
            and current_agent_step - bank['opp_started_step'] >= opp_timeout)
        tag_value = b + 1

        if bank['pending_opp_path'] is not None:
            if backend.count_aligned(pufferl, tag_value, 0) >= bank['num_hist_envs']:
                load_path = bank['pending_opp_path']
                backend.load_frozen_bank(pufferl, b, load_path)
                backend.count_aligned(pufferl, tag_value, 1)
                bank['cur_opp_path'] = load_path
                bank['pending_opp_path'] = None
                bank['hist_score'] = 0.0
                bank['hist_n'] = 0.0
                bank['opp_started_step'] = current_agent_step
                bank['last_epochs_to_align'] = epoch - bank.get('epoch_armed', epoch)
        elif timed_out:
            # Local pool is only our own snapshots — no per-swap nbytes filter
            # (that spammed every rank on timeout when a file was briefly
            # missing/partial). External arch filter stays in setup only.
            usable = pool_state['pool']
            if not usable:
                continue
            draw_slot = current_agent_step // opp_timeout
            opp_entry = sample_opponent_reproducible(
                usable,
                pool_state.get('seed', 0),
                pool_state.get('rank', 0),
                draw_slot + b,
                **sample_kwargs)
            bank['pending_opp_path'] = opp_entry['path']
            bank['epoch_armed'] = epoch
            bank['last_winrate_at_swap'] = winrate if winrate is not None else 0.0
            backend.count_aligned(pufferl, tag_value, 1)

    flat_logs['pool/size'] = len(pool_state['pool'])
    flat_logs['pool/num_banks'] = num_banks
    flat_logs['pool/sample'] = (
        1.0 if str(pool_state.get('sample', 'uniform')).startswith('p') else 0.0)
    total_score = 0.0
    total_n = 0.0
    for b in range(num_banks):
        bank = pool_state['banks'][b]
        wr = (bank['hist_score'] / bank['hist_n']
              if bank['hist_n'] > 0 else None)
        flat_logs[f'pool/winrate_at_swap_bank_{b}'] = bank['last_winrate_at_swap']
        flat_logs[f'pool/epochs_to_align_bank_{b}'] = bank['last_epochs_to_align']
        if wr is not None:
            flat_logs[f'pool/winrate_bank_{b}'] = wr
            flat_logs[f'env/historical_winrate_bank_{b}'] = wr
        total_score += bank['hist_score']
        total_n += bank['hist_n']
    if total_n > 0:
        agg = total_score / total_n
        flat_logs['pool/winrate'] = agg
        flat_logs['env/historical_winrate'] = agg
    seen = 0
    wr_sum = 0.0
    for e in pool_state['pool']:
        wr = learner_winrate(e)
        if wr is not None:
            seen += 1
            wr_sum += wr
    flat_logs['pool/pfsp_seen'] = seen
    if seen > 0:
        flat_logs['pool/pfsp_mean_learner_wr'] = wr_sum / seen
