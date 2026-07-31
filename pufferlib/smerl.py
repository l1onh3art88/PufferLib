"""Minimal multi-mode conditioning for self-play training.

One shared policy, K persistent behavior modes. Each primary rollout row is
assigned a fixed mode z for the run. The policy is conditioned by:

    h = encoder(obs) + embed[z]

Embeds train with PPO (Adam, embed_lr). No discriminator, no diversity reward,
no action-distance loss, no competence gates affecting training.

Optional mid-train heldout vs scripted bots is metrics-only (default off:
eval_interval_steps = 0). Manual eval still uses --smerl.force-mode.

Requires selfplay.enabled and env_name == 'robocode'.
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
    """Initialize mode layout + native embeds. No-op unless smerl.enabled."""
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
            f'smerl only supports env_name=robocode (got {env_name!r})')
    if not args.get('selfplay', {}).get('enabled', 0):
        raise RuntimeError('smerl requires selfplay.enabled = 1')

    num_modes = int(cfg.get('num_modes', 8))
    if num_modes < 2:
        raise RuntimeError('smerl.num_modes must be >= 2')

    # Heldout bots optional — only needed if eval_interval_steps > 0.
    heldout_bots = _parse_bot_list(cfg.get('heldout_bots', ''))
    eval_interval_steps = int(cfg.get('eval_interval_steps', 0))
    if eval_interval_steps > 0 and not heldout_bots:
        raise RuntimeError(
            "smerl.eval_interval_steps > 0 requires heldout_bots "
            "(e.g. heldout_bots = '3,4,5')")

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
    _ = rank  # reserved for multi-GPU layout

    # One mode per primary physical row. Frozen bank rows stay at sentinel -1.
    modes = np.full(total_agents, -1, dtype=np.int32)
    for b in range(num_buffers):
        start = b * agents_per_buffer
        row_modes = (np.arange(primary_per_buffer, dtype=np.int32) + b) % num_modes
        modes[start:start + primary_per_buffer] = row_modes
    backend.set_smerl_modes(pufferl, modes)

    # Gates unused in minimal mode; keep zeros for native buffer shape.
    gates = np.zeros(num_modes, dtype=np.int32)
    backend.set_smerl_gates(pufferl, gates)

    work_dir = os.path.join(args['checkpoint_dir'], args['env_name'], run_id, 'smerl')
    if artifact_owner:
        os.makedirs(work_dir, exist_ok=True)

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
        'eval_interval_steps': eval_interval_steps,
        'eval_games': int(cfg.get('eval_games', 4096)),
        'score_ema': float(cfg.get('score_ema', 0.8)),
        'heldout_bots': heldout_bots,
        'work_dir': work_dir,
        'world_size': max(1, int(args.get('world_size', 1))),
        'primary_per_buffer': primary_per_buffer,
        'agents_per_buffer': agents_per_buffer,
        'num_buffers': num_buffers,
        'last_logs': {'smerl/num_modes': float(num_modes)},
    }
    return state


def assign_modes(pufferl, backend, state):
    """Re-push the stored mode assignment (e.g. after force_mode eval)."""
    if state is None:
        return
    backend.set_smerl_modes(pufferl, state['modes'])


def update_gates(scores, gates, activation_score, enter_eps, exit_eps):
    """Legacy pure helper (tests). Gates do not affect minimal training."""
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
        eval_args.setdefault('smerl', {})
        eval_args['smerl'] = dict(eval_args.get('smerl') or {})
        eval_args['smerl']['enabled'] = 1
        eval_args['smerl']['force_mode'] = int(mode)
        eval_args.setdefault('selfplay', {})['enabled'] = 0
        eval_args['skip_match_close'] = False
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
    """Optional metrics-only heldout (does not change training)."""
    if state is None or not state.get('artifact_owner', True):
        return
    if not state.get('heldout_bots'):
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

    # Gates logged for dashboards only — not pushed as training signals.
    state['gates'] = update_gates(
        state['scores'], state['gates'],
        state['activation_score'],
        state['gate_enter_epsilon'],
        state['gate_exit_epsilon'],
    )
    assign_modes(pufferl, backend, state)

    best = float(np.max(state['scores']))
    state['last_logs'] = {
        'smerl/num_modes': float(K),
        'smerl/best_heldout': best,
    }
    for z in range(K):
        state['last_logs'][f'smerl/heldout_score_mode_{z}'] = float(state['scores'][z])
        state['last_logs'][f'smerl/raw_heldout_mode_{z}'] = float(raw[z])


def step(pufferl, backend, args, state, flat_logs):
    """Optional periodic heldout metrics. Merges smerl/* into flat_logs."""
    if state is None:
        return

    if state.get('last_logs'):
        flat_logs.update(state['last_logs'])
    else:
        flat_logs.setdefault('smerl/num_modes', float(state.get('num_modes', 0)))

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
