"""Backtest a chess policy against Maia (LC0 with Maia weights).

For each .bin checkpoint in --models-folder, runs --num-games games against
Maia and reports win rate + Elo estimate, then plots win rate vs training step.

Maia is launched by the C side (chess.h / CHESS_MODE_MAIA = 4) as a per-env
lc0 subprocess. We just set environment variables; the env's c_step pipes
FEN -> lc0 -> bestmove each opponent turn.

Requires `bash build.sh chess` after the latest chess.h changes.

Example:
    python maia_eval.py \\
        --models-folder checkpoints/chess/<run_id> \\
        --lc0-path ./lc0 \\
        --weights-path lc0/maia-1100.pb.gz \\
        --maia-elo 1100 \\
        --nodes 1 \\
        --num-games 200
"""
import argparse
import glob
import math
import os
import re
import sys
from copy import deepcopy

import matplotlib.pyplot as plt

from pufferlib import _C as backend
from pufferlib.pufferl import load_config, unroll_nested_dict


CHESS_MODE_MAIA = 4


def extract_step(path):
    """Pull a training step count from the filename. pufferl saves checkpoints
    as 16-digit zero-padded step numbers (e.g. '0000088605196288.bin' = 88.6B
    steps). Falls back to any leading digits, or 0 if none — so ad-hoc names
    like 275b_chess.bin still sort + evaluate."""
    name = os.path.basename(path)
    m = re.search(r'(\d+)\.bin$', name)
    if m:
        return int(m.group(1))
    m = re.match(r'(\d+)', name)
    if m:
        return int(m.group(1))
    return 0


def fmt_steps(n):
    """Compact human-readable step count for axis labels."""
    if n >= 1_000_000_000:
        return f'{n / 1e9:.1f}B'
    if n >= 1_000_000:
        return f'{n / 1e6:.1f}M'
    if n >= 1_000:
        return f'{n / 1e3:.1f}K'
    return f'{n:d}'


def eval_one(model_path, num_games, total_agents, num_threads, verbose):
    """Run num_games games of the loaded policy vs Maia, return win rate."""
    args = load_config('chess')
    # Force single-agent (learner only) mode with Maia as opponent.
    args['env']['mode'] = CHESS_MODE_MAIA
    # Disable the FEN curriculum so every game starts from the standard chess
    # starting position. Training defaults to fen_curric_pct=0.9, which would
    # otherwise have 90% of eval games start from random opening positions —
    # incomparable to Maia's calibration (human games from move 1) and
    # introduces huge variance from biased starting conditions.
    args['env']['fen_curric_pct'] = 0.0
    args['env']['random_fen'] = 0
    args['vec']['total_agents'] = int(total_agents)
    args['vec']['num_buffers'] = 1
    args['vec']['num_threads'] = int(num_threads)
    args['vec']['num_frozen_banks'] = 0
    args['vec']['frozen_bank_pct'] = 0.0
    args['selfplay']['enabled'] = 0
    args['train']['horizon'] = 1
    args['train']['minibatch_size'] = 1
    args['reset_state'] = False
    args['rank'] = 0
    args['gpu_id'] = 0
    args['nccl_id'] = b''
    args['world_size'] = 1
    args['wandb'] = False
    args['tag'] = None

    pufferl = backend.create_pufferl(args)
    backend.load_weights(pufferl, model_path)

    score_sum = 0.0
    n_total = 0
    last_print = -1
    logs = {}
    while n_total < num_games:
        backend.rollouts(pufferl)
        logs = dict(unroll_nested_dict(backend.eval_log(pufferl)))
        n = float(logs.get('env/n', 0.0))
        score = float(logs.get('env/score', 0.0))
        if n > 0:
            score_sum = score * n
            n_total = int(n)
            # Print only when at least one new game has finished, so we don't
            # spam identical lines on rollouts where no game terminated.
            if verbose and n_total != last_print:
                print(f'    games={n_total}/{num_games}  score={score:.3f}')
                last_print = n_total

    # Recover raw per-color counts: the aggregator divides Log fields by total n.
    n_f = float(n_total)
    games_w = float(logs.get('env/games_as_white', 0.0)) * n_f
    games_b = float(logs.get('env/games_as_black',  0.0)) * n_f
    score_w = float(logs.get('env/wins_as_white',   0.0)) * n_f  # win=1, draw=0.5
    score_b = float(logs.get('env/wins_as_black',   0.0)) * n_f
    failures = float(logs.get('env/maia_failures', 0.0)) * n_f
    draw_rate = float(logs.get('env/draw_rate', 0.0))  # already a rate

    backend.close(pufferl)
    score_rate = score_sum / max(n_total, 1)
    # score = wins + 0.5*draws  →  pure wins = score - 0.5*draws
    pure_win_rate = max(0.0, score_rate - 0.5 * draw_rate)
    loss_rate = max(0.0, 1.0 - pure_win_rate - draw_rate)
    return {
        'score_rate': score_rate,         # = win + 0.5*draw, used for Elo
        'pure_win_rate': pure_win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'games': n_total,
        'score_rate_white': (score_w / games_w) if games_w > 0 else None,
        'score_rate_black': (score_b / games_b) if games_b > 0 else None,
        'games_white': int(round(games_w)),
        'games_black': int(round(games_b)),
        'maia_failures': int(round(failures)),
    }


def winrate_to_elo(winrate, opponent_elo):
    wr = max(0.001, min(0.999, winrate))
    elo_diff = -400.0 * math.log10((1.0 - wr) / wr)
    return opponent_elo + elo_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models-folder', type=str, required=True,
                   help='Folder containing .bin checkpoints')
    p.add_argument('--lc0-path', type=str,
                   default='/puffertank/pufferlib/lc0/build/release/lc0')
    p.add_argument('--weights-path', type=str,
                   default='/puffertank/pufferlib/lc0/maia-1100.pb.gz')
    p.add_argument('--maia-backend', type=str, default='cuda-fp16',
                   help='lc0 --backend value (cuda-fp16=GPU, eigen=CPU). '
                        'First cuda run JIT-compiles kernels (~30-60s); cached after.')
    p.add_argument('--maia-elo', type=int, default=1100,
                   help='Approximate Maia rating for Elo estimation '
                        '(match the weights file, e.g. 1100 for maia-1100)')
    p.add_argument('--nodes', type=int, default=1,
                   help='lc0 search nodes per move (1 = weakest)')
    p.add_argument('--num-games', type=int, default=200)
    p.add_argument('--total-agents', type=int, default=64,
                   help='Number of parallel chess envs / lc0 subprocesses')
    p.add_argument('--num-threads', type=int, default=1,
                   help='OMP threads. Keep low when forking many subprocesses.')
    p.add_argument('--output-plot', type=str, default='maia_winrate.png')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    # load_config() in pufferl re-parses sys.argv with its own argparse and
    # blows up on our flags. Strip them now that we've consumed ours.
    sys.argv = [sys.argv[0]]

    # Wire Maia config into C via env vars (read by chess.h maia_init).
    os.environ['MAIA_LC0_PATH'] = args.lc0_path
    os.environ['MAIA_WEIGHTS_PATH'] = args.weights_path
    os.environ['MAIA_NODES'] = str(args.nodes)
    if args.maia_backend:
        os.environ['MAIA_BACKEND'] = args.maia_backend

    bin_files = glob.glob(os.path.join(args.models_folder, '**', '*.bin'),
                          recursive=True)
    # Skip pool/ subdirs (selfplay opponents) when scanning a run directory.
    # If the user explicitly pointed at a pool/ folder, keep those files —
    # that's an intentional choice to evaluate the selfplay-pool progression.
    folder_basename = os.path.basename(os.path.normpath(args.models_folder))
    if folder_basename != 'pool':
        bin_files = [f for f in bin_files if '/pool/' not in f and '/pool' not in os.path.dirname(f)]
    pairs = [(extract_step(f), f) for f in bin_files]
    pairs.sort()
    if not pairs:
        print(f'No .bin checkpoints found under {args.models_folder}')
        return

    print(f'Found {len(pairs)} checkpoints, running {args.num_games} games each')
    print(f'Maia config: lc0={args.lc0_path}  weights={args.weights_path}  '
          f'nodes={args.nodes}  backend={args.maia_backend or "default"}')

    steps, win_rates, elos = [], [], []
    for step, path in pairs:
        print(f'\n=== steps={fmt_steps(step)} ({step})  {path} ===')
        result = eval_one(
            model_path=path,
            num_games=args.num_games,
            total_agents=args.total_agents,
            num_threads=args.num_threads,
            verbose=args.verbose,
        )
        score = result['score_rate']
        elo = winrate_to_elo(score, args.maia_elo)
        wr_w = result['score_rate_white']
        wr_b = result['score_rate_black']
        wr_w_str = f'{wr_w:.2%}' if wr_w is not None else 'n/a'
        wr_b_str = f'{wr_b:.2%}' if wr_b is not None else 'n/a'
        print(f'    score={score:.2%}  win={result["pure_win_rate"]:.2%}  '
              f'draw={result["draw_rate"]:.2%}  loss={result["loss_rate"]:.2%}  '
              f'est_elo={elo:.0f}  games={result["games"]}')
        print(f'    score as white: {wr_w_str} ({result["games_white"]} games)  '
              f'as black: {wr_b_str} ({result["games_black"]} games)')
        if result['maia_failures'] > 0:
            print(f'    WARNING: maia_failures={result["maia_failures"]} '
                  f'(random-move fallback fired this many times - eval is degraded)')
        steps.append(step)
        win_rates.append(score)
        elos.append(elo)

    # X-axis: training-step counts parsed from the .bin filenames (see
    # extract_step). For checkpoints saved as 16-digit zero-padded step ints,
    # these will run up into the tens-to-hundreds of billions; format with
    # B/M/K suffixes so the axis stays readable.
    step_fmt = plt.FuncFormatter(lambda x, _: fmt_steps(int(x)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(steps, win_rates, marker='o')
    ax1.set_xlabel('Training steps')
    ax1.set_ylabel('Score rate vs Maia')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.xaxis.set_major_formatter(step_fmt)
    ax1.set_title(f'Score rate vs Maia ({args.maia_elo} Elo, nodes={args.nodes})')
    ax2.plot(steps, elos, marker='s', color='green')
    ax2.axhline(y=args.maia_elo, color='red', linestyle='--', alpha=0.7,
                label=f'Maia ({args.maia_elo})')
    ax2.set_xlabel('Training steps')
    ax2.set_ylabel('Estimated Elo')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(step_fmt)
    ax2.legend()
    ax2.set_title('Estimated Elo')
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f'\nSaved plot to {args.output_plot}')

    print(f'\n{"steps":>14}  {"score":>10}  {"est_elo":>8}')
    for s, wr, e in zip(steps, win_rates, elos):
        print(f'{fmt_steps(s):>14}  {wr:>9.2%}  {e:>8.0f}')


if __name__ == '__main__':
    main()
