"""Train the learned bot against the base search, then against itself.

The network plays the games. At every position the base search says what it
would have played, and that move is the policy target. The value target is
who eventually won. 9×9 is the main lesson; the same weights then play 15×15.

Run from the project directory:

    python train.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from engine.board import BLACK, WHITE, Board
from engine.learned import select_action
from engine.net import PolicyValueNet, encode, save_checkpoint
from engine.search import best_move

# name, size, mode, games, epochs, learning rate
PHASES = [
    ("9×9 demonstrations", 9, "demonstrate", 60, 8, 1e-3),
    ("15×15 demonstrations", 15, "demonstrate", 80, 10, 5e-4),
    ("15×15 against the base", 15, "dagger", 16, 2, 1e-4),
    ("15×15 self-play", 15, "self", 16, 2, 1e-4),
]
BASE_DEPTH = 2
BASE_TIME = 0.1
BATCH = 64


def main() -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    torch.set_num_threads(4)
    net = PolicyValueNet()
    opt = torch.optim.AdamW(net.parameters(), lr=PHASES[0][5], weight_decay=0.0)
    started = time.perf_counter()
    for name, size, mode, games, epochs, lr in PHASES:
        for group in opt.param_groups:
            group["lr"] = lr
        print(f"\n{name}: {games} games", flush=True)
        samples = collect(net, size, mode, games)
        loss, accuracy = fit(net, opt, samples, epochs)
        print(
            f"  examples {len(samples)}  loss {loss:.3f}  "
            f"agrees with the base on {accuracy:.0%} of them",
            flush=True,
        )
    wins, losses, draws, agreement = evaluate(net)
    save_checkpoint(net, wins=wins, losses=losses, draws=draws, agreement=agreement)
    elapsed = time.perf_counter() - started
    print(
        f"\nGreedy learned bot vs the base on 15×15: "
        f"{wins} wins, {losses} losses, {draws} draws "
        f"(same move on {agreement:.0%} of turns).",
        flush=True,
    )
    print(f"Saved models/learned.pt in {elapsed:.0f}s.", flush=True)


def collect(net, size, mode, games):
    samples = []
    for game in range(games):
        learner = BLACK if game % 2 == 0 else WHITE
        samples.extend(play_game(net, size, mode, learner))
        if (game + 1) % 10 == 0:
            print(f"  game {game + 1}/{games}", flush=True)
    return samples


def play_game(net, size, mode, learner):
    board = Board(size=size)
    played = []
    while not board.winner and not board.is_full() and board.legal_moves():
        side = BLACK if len(board.history) % 2 == 0 else WHITE
        teacher = best_move(board, side, time_limit=BASE_TIME, max_depth=BASE_DEPTH)
        if teacher is None:
            break
        row, col = _played_move(net, board, side, mode, learner, teacher)
        legal = np.zeros(size * size, np.bool_)
        for legal_row, legal_col in board.legal_moves():
            legal[legal_row * size + legal_col] = True
        played.append(
            {
                "planes": encode(board, side),
                "action": teacher.row * size + teacher.col,
                "legal": legal,
                "side": side,
            }
        )
        board.place(row, col, side)
    for sample in played:
        if board.winner == 0:
            sample["z"] = 0.0
        elif board.winner == sample["side"]:
            sample["z"] = 1.0
        else:
            sample["z"] = -1.0
    return played


def _played_move(net, board, side, mode, learner, teacher):
    if mode == "demonstrate":
        return teacher.row, teacher.col
    explore = mode == "dagger" and side == learner and np.random.random() < 0.5
    if mode == "self" or explore:
        row, col, _source, _value = select_action(net, board, side, temperature=0.3)
        return row, col
    return teacher.row, teacher.col


def fit(net, opt, samples, epochs):
    if not samples:
        return 0.0, 0.0
    net.train()
    last_loss = 0.0
    last_accuracy = 0.0
    order = np.arange(len(samples))
    for _ in range(epochs):
        np.random.shuffle(order)
        correct = 0
        seen = 0
        for start in range(0, len(order), BATCH):
            batch = [samples[int(i)] for i in order[start : start + BATCH]]
            last_loss, hit, count = step(net, opt, batch)
            correct += hit
            seen += count
        last_accuracy = correct / seen if seen else 0.0
    net.eval()
    return last_loss, last_accuracy


def step(net, opt, batch):
    planes = torch.from_numpy(np.stack([item["planes"] for item in batch]))
    actions = torch.tensor([item["action"] for item in batch], dtype=torch.long)
    legal = torch.from_numpy(np.stack([item["legal"] for item in batch]))
    outcome = torch.tensor([item["z"] for item in batch], dtype=torch.float32)
    logits, values = net(planes)
    logits = logits.masked_fill(~legal, -1e9)
    logp = torch.log_softmax(logits, dim=-1)
    chosen = logp[torch.arange(len(batch)), actions]
    loss = (-chosen).mean() + 0.1 * nn.functional.mse_loss(values, outcome)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    hits = int((logits.argmax(dim=-1) == actions).sum().item())
    return float(loss.item()), hits, len(batch)


def evaluate(net):
    wins = losses = draws = 0
    same = 0
    turns = 0
    for game in range(8):
        learner = BLACK if game % 2 == 0 else WHITE
        board = Board(size=15)
        while not board.winner and not board.is_full() and board.legal_moves():
            side = BLACK if len(board.history) % 2 == 0 else WHITE
            teacher = best_move(board, side, time_limit=BASE_TIME, max_depth=BASE_DEPTH)
            if teacher is None:
                break
            if side == learner:
                row, col, _source, _value = select_action(net, board, side, temperature=0.0)
                same += int((row, col) == (teacher.row, teacher.col))
                turns += 1
                board.place(row, col, side)
            else:
                board.place(teacher.row, teacher.col, side)
        if board.winner == learner:
            wins += 1
        elif board.winner == 0:
            draws += 1
        else:
            losses += 1
    agreement = same / turns if turns else 0.0
    return wins, losses, draws, agreement


if __name__ == "__main__":
    main()
