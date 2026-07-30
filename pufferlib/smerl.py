"""SMERL-like mode diversity for self-play training.

One policy, K persistent behavior modes. Modes are a property of primary-policy
rollout rows (fixed for the run). A discriminator is trained to predict z from
pre-mode encoder features, and a small diversity bonus is paid only to modes
that clear heldout competence gates.

Robocode adaptation (v1 target):
  - 1v1, so each primary agent row gets its own mode (no team sharing).
  - Heldout score is winrate vs scripted eval bots (`puffer eval_bot`), not
    vs a frozen checkpoint anchor.
  - Default-off; requires selfplay.enabled and env_name == 'robocode'.
"""
from __future__ import annotations

import os
from copy import deepcopy

import numpy as np

from pufferlib import _C


def _parse_bot_list(spec):
    """Parse heldout_bots config: '3,4,5' or [3,4,5] or a single int."""
    if spec is None or spec == '' or spec == 0:
        return []
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    if isinstance(spec, (int, float)):
        return [int(spec)]
    text = str(spec).strip().strip("'\"")
    if not text:
        return []
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _cfg(args):
    return args.get('smerl', {}) or {}


def enabled(args):
    return bool(_cfg(args).get('enabled', 0))


def setup(pufferl, backend, args, run_id, artifact_owner=True, pool_state=None):
    """Initialize modes/gates and native buffers. No-op unless smerl.enabled."""
    cfg = _cfg(args)
    if not cfg.get('enabled', 0):
        return None
    if backend is not _C:
        raise RuntimeError('smerl requires the native CUDA backend')
    if not backend.smerl_enabled(pufferl):
        raise RuntimeError('smerl.enabled in config but native SMERL was not created')

    env_name = args.get('env_name')
    if env_name != 'robocode':
        raise RuntimeError(
            f'smerl v1 only supports env_name=robocode (got {env_name!r})')
    if not args.get('selfplay', {}).get('enabled', 0):
        raise RuntimeError('smerl requires selfplay.enabled = 1')

    num_modes = int(cfg.get('num_modes', 8))
    if num_modes < 2:
        raise RuntimeError('smerl.num_modes must be >= 2')

    heldout_bots = _parse_bot_list(cfg.get('heldout_bots', '3'))
    if not heldout_bots:
        raise RuntimeError(
            "smerl.heldout_bots is empty — e.g. heldout_bots = '3,4,5' "
            '(wave_surfer, hawk_on_fire, raiko)')

    total_agents = int(args['vec']['total_agents'])
    num_buffers = int(args['vec']['num_buffers'])
    if total_agents % num_buffers != 0:
        raise RuntimeError('total_agents must be divisible by num_buffers')
    agents_per_buffer = total_agents // num_buffers

    num_envs = backend.num_envs(pufferl)
    agents_per_env = total_agents // num_envs
    if agents_per_env % 2 != 0:
        raise RuntimeError('agents_per_env must be even (two equal sides)')
    team_size = agents_per_env // 2

    num_banks = int(args['vec'].get('num_frozen_banks', 1))
    frozen_size = int(agents_per_buffer * float(args['vec'].get('frozen_bank_pct', 0.0)))
    frozen_size -= frozen_size % team_size
    total_frozen = frozen_size * max(num_banks, 0)
    primary_per_buffer = agents_per_buffer - total_frozen
    if primary_per_buffer <= 0:
        raise RuntimeError('smerl: no primary rows left after frozen banks')

    rank = int(args.get('rank', 0))
    # Deterministic mode layout — no RNG. Numpy Generator.shuffle is seeded but
    # still a footgun (version drift, accidental unseeded paths). Stripe modes
    # across primary rows; offset by buffer index so buffers are not identical.
    # Matches "fixed z per row for the run" and is bit-stable given layout.
    _ = rank  # reserved if we ever need rank-striped layouts in multi-GPU

    # One mode per primary physical row. Frozen bank rows stay at sentinel -1
    # (native leave them unconditioned and never pays them a bonus).
    modes = np.full(total_agents, -1, dtype=np.int32)
    for b in range(num_buffers):
        start = b * agents_per_buffer
        row_modes = (np.arange(primary_per_buffer, dtype=np.int32) + b) % num_modes
        modes[start:start + primary_per_buffer] = row_modes
    backend.set_smerl_modes(pufferl, modes)

    gates = np.zeros(num_modes, dtype=np.int32)
    backend.set_smerl_gates(pufferl, gates)

    work_dir = os.path.join(args['checkpoint_dir'], args['env_name'], run_id, 'smerl')
    if artifact_owner:
        os.makedirs(work_dir, exist_ok=True)

    # Optional resume from a sidecar next to a loaded checkpoint.
    load_path = args.get('load_model_path')
    if load_path and load_path != 'latest':
        sidecar = load_path + '.smerl'
        if os.path.isfile(sidecar):
            backend.load_smerl(pufferl, sidecar)

    state = {
        'enabled': True,
        'artifact_owner': artifact_owner,
        'num_modes': num_modes,
        'modes': modes,
        'gates': gates.copy(),
        'scores': np.zeros(num_modes, dtype=np.float64),
        'score_initialized': np.zeros(num_modes, dtype=np.bool_),
        'last_eval_step': 0,
        'activation_score': float(cfg.get('activation_score', 0.25)),
        'gate_enter_epsilon': float(cfg.get('gate_enter_epsilon', 0.03)),
        'gate_exit_epsilon': float(cfg.get('gate_exit_epsilon', 0.06)),
        'eval_interval_steps': int(cfg.get('eval_interval_steps', 200_000_000)),
        'eval_games': int(cfg.get('eval_games', 4096)),
        'score_ema': float(cfg.get('score_ema', 0.8)),
        'heldout_bots': heldout_bots,
        'work_dir': work_dir,
        'world_size': max(1, int(args.get('world_size', 1))),
        'primary_per_buffer': primary_per_buffer,
        'agents_per_buffer': agents_per_buffer,
        'num_buffers': num_buffers,
        # Cached logs from the last gating pass (merged into flat_logs by step).
        'last_logs': {},
    }
    return state


def assign_modes(pufferl, backend, state):
    """Re-push the stored mode assignment (e.g. after force_mode eval)."""
    if state is None:
        return
    backend.set_smerl_modes(pufferl, state['modes'])


def update_gates(scores, gates, activation_score, enter_eps, exit_eps):
    """Two-stage heldout gating with hysteresis.

    Returns a new int32 gate array (0/1). Pure function for unit testing.
    """
    scores = np.asarray(scores, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.int32).copy()
    K = len(scores)
    assert len(gates) == K

    best = float(np.max(scores)) if K else 0.0
    if best < activation_score:
        gates[:] = 0
        return gates

    enter_thresh = best - enter_eps
    exit_thresh = best - exit_eps
    for z in range(K):
        if gates[z]:
            if scores[z] < exit_thresh:
                gates[z] = 0
        else:
            if scores[z] >= enter_thresh:
                gates[z] = 1
    return gates


def _heldout_eval_mode(env_name, policy_path, mode, bots, eval_games, base_args, verbose=False):
    """Score one mode against each heldout bot; return mean winrate."""
    from pufferlib.pufferl import eval_bot

    winrates = []
    for bot in bots:
        eval_args = deepcopy(base_args)
        eval_args['env_name'] = env_name
        eval_args.setdefault('env', {})
        eval_args['env']['bot_policy'] = int(bot)
        # Keep SMERL on so force_mode / sidecar load work; disable selfplay.
        eval_args.setdefault('smerl', {})
        eval_args['smerl'] = dict(eval_args.get('smerl') or {})
        eval_args['smerl']['enabled'] = 1
        eval_args['smerl']['force_mode'] = int(mode)
        # Inherit num_modes etc. from training config so native arch matches.
        eval_args.setdefault('selfplay', {})['enabled'] = 0
        eval_args['skip_match_close'] = False
        # Quiet heldout by default — training logs already show gate updates.
        logs = eval_bot(
            env_name,
            policy_path=policy_path,
            num_games=eval_games,
            bot_policy=int(bot),
            args=eval_args,
            verbose=verbose,
        )
        winrates.append(float(logs.get('env/slot_0_score', 0.0)))
    return float(np.mean(winrates)) if winrates else 0.0


def run_heldout_eval(pufferl, backend, args, state, verbose=False):
    """Evaluate every mode vs heldout bots, update EMA scores and gates."""
    if state is None or not state.get('artifact_owner', True):
        return

    env_name = args['env_name']
    work_dir = state['work_dir']
    os.makedirs(work_dir, exist_ok=True)
    policy_path = os.path.join(work_dir, 'heldout_eval.bin')
    backend.save_weights(pufferl, policy_path)
    backend.save_smerl(pufferl, policy_path + '.smerl')

    K = state['num_modes']
    ema = state['score_ema']
    raw = np.zeros(K, dtype=np.float64)
    for z in range(K):
        raw[z] = _heldout_eval_mode(
            env_name, policy_path, z, state['heldout_bots'],
            state['eval_games'], args, verbose=verbose)
        if state['score_initialized'][z]:
            state['scores'][z] = ema * state['scores'][z] + (1.0 - ema) * raw[z]
        else:
            state['scores'][z] = raw[z]
            state['score_initialized'][z] = True

    state['gates'] = update_gates(
        state['scores'], state['gates'],
        state['activation_score'],
        state['gate_enter_epsilon'],
        state['gate_exit_epsilon'],
    )
    backend.set_smerl_gates(pufferl, state['gates'])
    # Restore training mode assignment (force_mode only touched the eval pufferl,
    # but be explicit in case a future path reuses the train instance).
    assign_modes(pufferl, backend, state)

    best = float(np.max(state['scores']))
    state['last_logs'] = {
        'smerl/active': 1.0 if best >= state['activation_score'] else 0.0,
        'smerl/gated_modes': float(np.sum(state['gates'])),
        'smerl/best_heldout': best,
    }
    for z in range(K):
        state['last_logs'][f'smerl/heldout_score_mode_{z}'] = float(state['scores'][z])
        state['last_logs'][f'smerl/gate_mode_{z}'] = float(state['gates'][z])
        state['last_logs'][f'smerl/raw_heldout_mode_{z}'] = float(raw[z])


def step(pufferl, backend, args, state, flat_logs):
    """Periodic heldout eval + gate update. Merges smerl/* into flat_logs."""
    if state is None:
        return

    # Always surface last known gate state + native disc metrics.
    if state.get('last_logs'):
        flat_logs.update(state['last_logs'])
    else:
        flat_logs.setdefault('smerl/active', 0.0)
        flat_logs.setdefault('smerl/gated_modes', 0.0)

    if not state.get('artifact_owner', True):
        return

    world_size = int(state.get('world_size', 1))
    agent_steps = int(pufferl.global_step) * world_size
    interval = int(state['eval_interval_steps'])
    if interval <= 0:
        return
    if agent_steps - int(state['last_eval_step']) < interval:
        return

    run_heldout_eval(pufferl, backend, args, state, verbose=False)
    state['last_eval_step'] = agent_steps
    flat_logs.update(state['last_logs'])


def save_sidecar(pufferl, backend, model_path):
    if backend is not _C or not backend.smerl_enabled(pufferl):
        return
    backend.save_smerl(pufferl, model_path + '.smerl')


def load_sidecar(pufferl, backend, model_path):
    if backend is not _C or not backend.smerl_enabled(pufferl):
        return
    sidecar = model_path + '.smerl'
    if os.path.isfile(sidecar):
        backend.load_smerl(pufferl, sidecar)
