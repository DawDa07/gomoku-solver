"""Fine-tune the network to stretch games against the solved Black line.

The book already labels every White reply with how many moves Black's win
still takes. The policy target is the reply or replies with the largest
count. Those examples do not train the value head. A sample of the self-play
replay is mixed in, so positions outside the book still have a win/loss
signal.

    python tools/train_delay.py

The run saves models/delay.pt, then plays the solved line as Black against
that net as White from the center and prints the length in plies. The book
line reaches 35.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.board import WHITE, Board  # noqa: E402
from engine.book import iter_white_targets  # noqa: E402
from engine.learned import BEST, DELAY  # noqa: E402
from engine.net import (  # noqa: E402
    CHECKPOINT,
    PolicyValueNet,
    encode,
    load_checkpoint,
    save_checkpoint,
)
from selfplay import REPLAY, _symmetry  # noqa: E402

BATCH = 64
EPOCHS = 3
LR = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    torch.manual_seed(1)
    np.random.seed(1)
    torch.set_num_threads(4)

    delay = collect_delay()
    print(f"Book positions for White: {len(delay)} orientations", flush=True)
    net = load_checkpoint(torch.device("cpu"), BEST)
    if net is None:
        net = load_checkpoint(torch.device("cpu"), CHECKPOINT)
    if net is None:
        print("No checkpoint to fine-tune. Starting from random weights.", flush=True)
        net = PolicyValueNet()
    else:
        print(f"Fine-tuning from {BEST.name if BEST.is_file() else CHECKPOINT.name}.", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    for epoch in range(args.epochs):
        mixed = delay + replay_batch(np.random.default_rng(epoch))
        loss, accuracy = fit(net, opt, mixed, np.random.default_rng(epoch + 100))
        print(f"  epoch {epoch + 1}/{args.epochs}  loss {loss:.3f}  delay target {accuracy:.0%}", flush=True)
    save_checkpoint(net, DELAY)
    print(f"Saved {DELAY.relative_to(ROOT)}.", flush=True)
    plies, winner = measure()
    who = {0: "no result", 1: "Black won", 2: "White won"}[winner]
    print(f"From h8, solved Black against this net: {plies} plies, {who}. The book line reaches 35.", flush=True)


def collect_delay() -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Eight reflections of each White book position. Value weight is zero."""
    samples = []
    for history, targets in iter_white_targets():
        board = Board()
        for row, col, player in history:
            board.place(row, col, player)
        policy = np.zeros(board.size * board.size, np.float32)
        for row, col in targets:
            policy[row * board.size + col] = 1.0
        policy /= policy.sum()
        planes = encode(board, WHITE)
        for turn in range(8):
            turned, turned_policy = _symmetry(planes, policy, turn)
            samples.append((turned, turned_policy, 0.0, 0.0))
    return samples


def replay_batch(rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Self-play positions, one random reflection each. Value weight is one."""
    if not REPLAY.is_file():
        return []
    saved = np.load(REPLAY)
    if int(saved["size"]) != 15:
        return []
    planes, policies, values = saved["planes"], saved["policies"], saved["values"]
    samples = []
    for index in range(len(values)):
        turn = int(rng.integers(8))
        board, policy = _symmetry(planes[index], policies[index], turn)
        samples.append((board, policy, float(values[index]), 1.0))
    return samples


def fit(net, opt, samples, rng: np.random.Generator) -> tuple[float, float]:
    net.train()
    order = rng.permutation(len(samples))
    total_loss = 0.0
    hit = 0
    delay_seen = 0
    steps = 0
    for start in range(0, len(order), BATCH):
        batch = [samples[int(i)] for i in order[start : start + BATCH]]
        loss, batch_hit, batch_delay = step(net, opt, batch)
        total_loss += loss
        hit += batch_hit
        delay_seen += batch_delay
        steps += 1
    net.eval()
    accuracy = hit / delay_seen if delay_seen else 0.0
    return total_loss / max(steps, 1), accuracy


def step(net, opt, batch) -> tuple[float, int, int]:
    planes = torch.from_numpy(np.stack([item[0] for item in batch]))
    target = torch.from_numpy(np.stack([item[1] for item in batch]))
    outcome = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    weight = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    logits, values = net(planes)
    logp = torch.log_softmax(logits, dim=-1)
    policy_loss = -(target * logp).sum(dim=1).mean()
    if float(weight.sum()) > 0:
        value_loss = (((values - outcome) ** 2) * weight).sum() / weight.sum()
    else:
        value_loss = logits.new_zeros(())
    loss = policy_loss + 0.1 * value_loss
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    # Delay rows have a zero value weight. Count a hit when the top move is one of theirs.
    chosen = logits.argmax(dim=-1)
    delay = weight == 0
    if int(delay.sum()) == 0:
        return float(loss.item()), 0, 0
    picked = target[delay, chosen[delay]]
    return float(loss.item()), int((picked > 0).sum().item()), int(delay.sum())


def measure() -> tuple[int, int]:
    """Solved Black against the delay net, opening at the center."""
    from engine.players import make_player
    from tools.arena import play_game

    black = make_player("book")
    white = make_player(f"learned:path={DELAY}")
    try:
        game = play_game(black, white, [(7, 7)], 15, 0)
    finally:
        black.close()
        white.close()
    return len(game.moves), game.winner


if __name__ == "__main__":
    main()
