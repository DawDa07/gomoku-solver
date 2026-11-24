"""AlphaZero-style training: the network learns from its own tree search.

Each iteration:

1. The best network so far plays itself. Every move comes from a tree search
   with a fixed number of playouts. Early moves are sampled in proportion to
   the visit counts, later ones are the most visited, and the root gets some
   random noise so new moves get tried.
2. Every position is stored with two targets: how often the search visited
   each move, and who won the game. The newest positions replace the oldest
   in a replay buffer.
3. A candidate network trains on random batches from the buffer, each board
   turned and mirrored at random, which gives eight versions of each.
4. The candidate plays the best network in the tournament. It replaces
   models/best.pt only if it scores at least 55%.

    python selfplay.py --size 9 --iterations 10
    python selfplay.py --size 15 --iterations 2 --sims 100

The first run copies models/learned.pt to models/best.pt. The candidate
keeps training across iterations in models/candidate.pt, whether or not it
passes the gate. The log goes to results/selfplay.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import deque
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch import nn

from engine.board import BLACK, EMPTY, Board, opponent
from engine.mcts import MCTS, pick_device
from engine.net import CHECKPOINT, encode, load_checkpoint, save_checkpoint
from engine.threats import scan

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
BEST = MODELS / "best.pt"
CANDIDATE = MODELS / "candidate.pt"
LOG = ROOT / "results" / "selfplay.jsonl"

NOISE = 0.25
GATE_SCORE = 0.55


def _symmetry(planes: np.ndarray, policy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """One of the eight turns and mirrors of a board and its policy target."""
    size = planes.shape[-1]
    grid = policy.reshape(size, size)
    planes = np.rot90(planes, k % 4, axes=(1, 2))
    grid = np.rot90(grid, k % 4)
    if k >= 4:
        planes = planes[:, :, ::-1]
        grid = grid[:, ::-1]
    return np.ascontiguousarray(planes), np.ascontiguousarray(grid).reshape(-1)


def play_game(
    search: MCTS,
    size: int,
    simulations: int,
    sample_moves: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], int, int]:
    """One self-play game. Returns (planes, visit share, side to move) per position, the winner, and the length."""
    board = Board(size=size)
    side = BLACK
    mid = size // 2
    # A random first stone near the center, so games differ from the start.
    board.place(mid + int(rng.integers(-1, 2)), mid + int(rng.integers(-1, 2)), side)
    side = opponent(side)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    while not board.winner and EMPTY in board.cells:
        policy = np.zeros(size * size, np.float32)
        wins = scan(board, side)[0]
        their_wins = scan(board, opponent(side))[0]
        if wins or len(their_wins) == 1:
            move = min(wins or their_wins)
            policy[move] = 1.0
        else:
            moves, visits, _ = search.search(board, side, simulations, noise=NOISE, rng=rng)
            policy[moves] = visits / visits.sum()
            if len(board.history) < sample_moves:
                move = int(rng.choice(moves, p=visits / visits.sum()))
            else:
                move = int(moves[int(np.argmax(visits))])
        history.append((encode(board, side), policy, side))
        board.place(move // size, move % size, side)
        side = opponent(side)
    return history, board.winner, len(board.history)


def _init_worker() -> None:
    torch.set_num_threads(1)


def _worker(args: tuple[str, int, int, int, int, int]) -> tuple[list, list[tuple[int, int]]]:
    path, size, simulations, sample_moves, games, seed = args
    net = load_checkpoint(torch.device("cpu"), Path(path))
    assert net is not None
    search = MCTS(net, torch.device("cpu"))
    rng = np.random.default_rng(seed)
    samples = []
    outcomes = []
    for _ in range(games):
        history, winner, length = play_game(search, size, simulations, sample_moves, rng)
        for planes, policy, side in history:
            z = 0.0 if winner == 0 else (1.0 if winner == side else -1.0)
            samples.append((planes, policy, z))
        outcomes.append((winner, length))
    return samples, outcomes


def generate(path: Path, size: int, simulations: int, games: int, workers: int, seed: int):
    sample_moves = max(4, size // 2)
    per_task = 2
    tasks = [
        (str(path), size, simulations, sample_moves, per_task, seed * 10_000 + index)
        for index in range((games + per_task - 1) // per_task)
    ]
    samples: list = []
    outcomes: list = []
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for found, results in pool.imap_unordered(_worker, tasks):
            samples.extend(found)
            outcomes.extend(results)
    return samples, outcomes


def train(
    net: nn.Module,
    opt: torch.optim.Optimizer,
    buffer: deque,
    steps: int,
    batch: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[float, float]:
    net.train()
    policy_total = value_total = 0.0
    for _ in range(steps):
        picks = rng.integers(0, len(buffer), size=batch)
        planes, policies, values = [], [], []
        for index in picks:
            board, policy, z = buffer[index]
            board, policy = _symmetry(board, policy, int(rng.integers(8)))
            planes.append(board)
            policies.append(policy)
            values.append(z)
        x = torch.from_numpy(np.stack(planes)).to(device)
        target_policy = torch.from_numpy(np.stack(policies)).to(device)
        target_value = torch.tensor(values, dtype=torch.float32, device=device)
        logits, value = net(x)
        policy_loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = ((value - target_value) ** 2).mean()
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        policy_total += policy_loss.item()
        value_total += value_loss.item()
    net.eval()
    return policy_total / steps, value_total / steps


def gate(size: int, simulations: int, openings: int, workers: int) -> dict:
    from tools.arena import run_match

    candidate = f"mcts:sims={simulations},path={CANDIDATE.relative_to(ROOT)}"
    best = f"mcts:sims={simulations},path={BEST.relative_to(ROOT)}"
    return run_match(candidate, best, openings, size, workers, progress=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero-style self-play training.")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=128, help="self-play games per iteration")
    parser.add_argument("--sims", type=int, default=100, help="playouts per move in self-play")
    parser.add_argument("--buffer", type=int, default=60_000, help="positions kept for training")
    parser.add_argument("--steps", type=int, default=400, help="training batches per iteration")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-openings", type=int, default=20, help="tournament openings, each played twice")
    parser.add_argument("--gate-sims", type=int, default=100)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    MODELS.mkdir(exist_ok=True)
    LOG.parent.mkdir(exist_ok=True)
    if not BEST.exists():
        shutil.copy(CHECKPOINT, BEST)
        print(f"Starting from {CHECKPOINT.relative_to(ROOT)}.", flush=True)
    if not CANDIDATE.exists():
        shutil.copy(BEST, CANDIDATE)

    device = pick_device()
    net = load_checkpoint(device, CANDIDATE)
    assert net is not None
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    buffer: deque = deque(maxlen=args.buffer)

    for iteration in range(1, args.iterations + 1):
        started = time.perf_counter()
        samples, outcomes = generate(BEST, args.size, args.sims, args.games, args.workers, args.seed + iteration)
        buffer.extend(samples)
        played = time.perf_counter() - started
        black = sum(winner == BLACK for winner, _ in outcomes)
        draws = sum(winner == 0 for winner, _ in outcomes)
        mean_length = sum(length for _, length in outcomes) / len(outcomes)
        print(
            f"[{iteration}] {len(outcomes)} games on {args.size}×{args.size} in {played:.0f}s "
            f"({len(outcomes) / played * 3600:.0f} per hour), Black won {black}, {draws} draws, "
            f"{mean_length:.0f} moves on average; buffer {len(buffer)}",
            flush=True,
        )

        policy_loss, value_loss = train(net, opt, buffer, args.steps, args.batch, device, rng)
        save_checkpoint(net.to("cpu"), CANDIDATE, size=args.size)
        net.to(device)
        print(f"    trained {args.steps} batches: policy loss {policy_loss:.3f}, value loss {value_loss:.3f}", flush=True)

        result = gate(args.size, args.gate_sims, args.gate_openings, args.workers)
        accepted = result["score"] >= GATE_SCORE
        if accepted:
            shutil.copy(CANDIDATE, BEST)
        print(
            f"    candidate vs best: {result['score']:.0%} over {result['games']} games "
            f"({result['elo']:+.0f}) -> {'accepted as models/best.pt' if accepted else 'rejected'}",
            flush=True,
        )
        with LOG.open("a") as log:
            log.write(
                json.dumps(
                    {
                        "iteration": iteration,
                        "size": args.size,
                        "games": len(outcomes),
                        "games_per_hour": len(outcomes) / played * 3600,
                        "black_wins": black,
                        "draws": draws,
                        "mean_length": mean_length,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "gate_score": result["score"],
                        "accepted": accepted,
                        "seconds": time.perf_counter() - started,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    sys.exit(main())
