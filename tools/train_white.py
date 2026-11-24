"""Train the network White uses once a position is no longer a proved loss.

Two kinds of games, starting from models/best.pt:

- White is the network and Black is the base search.
- Black follows the solved book, except about one move in four is random.
  Off the stored tree, Black uses a short search so the game can finish.

The value target is who won, from the side to move. The policy target is
White's move in games White won. Losing moves train the value head only, so
they are not copied. Self-play replay is mixed in each epoch. Delay labels
are not: the book still supplies those at play time.

    python tools/train_white.py

Saves models/white.pt, then reports White's wins against the base bot and
the length of solved Black against this White from the center.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.board import BLACK, WHITE, Board  # noqa: E402
from engine.book import lookup, solved_move  # noqa: E402
from engine.learned import BEST, WHITE_NET, select_action, white_move  # noqa: E402
from engine.net import (  # noqa: E402
    PolicyValueNet,
    encode,
    load_checkpoint,
    save_checkpoint,
)
from engine.players import make_player  # noqa: E402
from engine.search import SearchResult, best_move  # noqa: E402

REPLAY = ROOT / "results" / "replay.npz"

BATCH = 64
EPOCHS = 3
GAMES = 8
LR = 1e-4
# A blunder this often is a person leaving the solved line.
BLUNDER = 0.25
MOVE_CAP = 120
BASE_TIME = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--games", type=int, default=GAMES)
    args = parser.parse_args()
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.set_num_threads(4)

    net = load_checkpoint(torch.device("cpu"), BEST)
    if net is None:
        print("No models/best.pt. Starting from random weights.", flush=True)
        net = PolicyValueNet()
    else:
        print("Fine-tuning from models/best.pt.", flush=True)
    played = collect(net, args.games)
    wins = sum(1 for item in played if item[2] > 0 and item[3] == 1.0)
    print(f"Collected {len(played)} positions, {wins} winning White moves.", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    for epoch in range(args.epochs):
        mixed = played + replay_batch(np.random.default_rng(epoch))
        loss, accuracy = fit(net, opt, mixed, np.random.default_rng(epoch + 100))
        print(f"  epoch {epoch + 1}/{args.epochs}  loss {loss:.3f}  policy {accuracy:.0%}", flush=True)
    save_checkpoint(net, WHITE_NET)
    print(f"Saved {WHITE_NET.relative_to(ROOT)}.", flush=True)
    base_wins, base_games, plies, winner = measure()
    who = {0: "no result", 1: "Black won", 2: "White won"}[winner]
    print(
        f"White won {base_wins} of {base_games} games against the base bot.",
        flush=True,
    )
    print(
        f"From h8, solved Black against this White: {plies} plies, {who}. The book line reaches 35.",
        flush=True,
    )


def collect(net, games: int) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Planes, policy, value, policy weight. Weight is 1 only for a move we want copied."""
    samples = []
    rng = random.Random(1)
    for index in range(games):
        samples.extend(play_base(net, rng))
        print(f"  base game {index + 1}/{games}", flush=True)
    for index in range(games):
        samples.extend(play_blunders(net, rng))
        print(f"  blunder game {index + 1}/{games}", flush=True)
    return samples


def play_base(net, rng: random.Random):
    black = make_player("base:time=0.1,depth=2")
    try:
        return _play(net, lambda board: black.move(board, BLACK), explore=True)
    finally:
        black.close()


def play_blunders(net, rng: random.Random):
    def black_move(board: Board) -> SearchResult | None:
        if board.history and rng.random() < BLUNDER:
            row, col = rng.choice(board.legal_moves())
            return SearchResult(row, col, 0, [(row, col)], "Blunder.")
        found = lookup(board, BLACK)
        if found.move is not None:
            return found.move
        proved = solved_move(board, BLACK)
        if proved is not None:
            return proved
        return best_move(board, BLACK, time_limit=BASE_TIME, max_depth=2)

    return _play(net, black_move, explore=True)


def _play(net, black_move, explore: bool):
    board = Board()
    side = BLACK
    played = []
    while not board.winner and not board.is_full() and len(board.history) < MOVE_CAP:
        if side == BLACK:
            result = black_move(board)
        else:
            temperature = 0.3 if explore else 0.0
            row, col, _reason, _value = select_action(net, board, WHITE, temperature)
            result = SearchResult(row, col, 0, [(row, col)], "")
        if result is None:
            break
        size = board.size
        policy = np.zeros(size * size, np.float32)
        policy[result.row * size + result.col] = 1.0
        played.append((encode(board, side), policy, side))
        board.place(result.row, result.col, side)
        side = WHITE if side == BLACK else BLACK
    winner = board.winner
    samples = []
    for planes, policy, moved in played:
        if winner == 0:
            value = 0.0
        elif winner == moved:
            value = 1.0
        else:
            value = -1.0
        # Copy White's move only from games White won. Losses train the value only.
        weight = 1.0 if winner == WHITE and moved == WHITE else 0.0
        samples.append((planes, policy, value, weight))
    return samples


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


def replay_batch(rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
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
    total = 0.0
    hit = seen = steps = 0
    for start in range(0, len(order), BATCH):
        batch = [samples[int(i)] for i in order[start : start + BATCH]]
        loss, batch_hit, batch_seen = step(net, opt, batch)
        total += loss
        hit += batch_hit
        seen += batch_seen
        steps += 1
    net.eval()
    accuracy = hit / seen if seen else 0.0
    return total / max(steps, 1), accuracy


def step(net, opt, batch) -> tuple[float, int, int]:
    planes = torch.from_numpy(np.stack([item[0] for item in batch]))
    target = torch.from_numpy(np.stack([item[1] for item in batch]))
    outcome = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    weight = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    logits, values = net(planes)
    logp = torch.log_softmax(logits, dim=-1)
    per_row = -(target * logp).sum(dim=1)
    if float(weight.sum()) > 0:
        policy_loss = (per_row * weight).sum() / weight.sum()
    else:
        policy_loss = logits.new_zeros(())
    value_loss = ((values - outcome) ** 2).mean()
    loss = policy_loss + 0.1 * value_loss
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    chosen = logits.argmax(dim=-1)
    copied = weight > 0
    if int(copied.sum()) == 0:
        return float(loss.item()), 0, 0
    picked = target[copied, chosen[copied]]
    return float(loss.item()), int((picked > 0).sum().item()), int(copied.sum())


def _play_out(black, white, opening: list[tuple[int, int]]):
    board = Board()
    side = BLACK
    for row, col in opening:
        board.place(row, col, side)
        side = WHITE if side == BLACK else BLACK
    while not board.winner and not board.is_full():
        player = black if side == BLACK else white
        result = player.move(board, side)
        if result is None:
            break
        board.place(result.row, result.col, side)
        side = WHITE if side == BLACK else BLACK
    return board


def measure() -> tuple[int, int, int, int]:
    """Wins against the base bot, games played, book-game length, book-game winner."""

    class _White:
        name = "white"

        def move(self, board, side):
            return white_move(board, side)

        def close(self):
            pass

    wins = 0
    games = 4
    white = _White()
    try:
        for index in range(games):
            black = make_player("base:time=0.1,depth=2")
            try:
                game = _play_out(black, white, [])
            finally:
                black.close()
            if game.winner == WHITE:
                wins += 1
            print(f"  base match {index + 1}/{games}: winner {game.winner}, {len(game.history)} plies", flush=True)
        book = make_player("book")
        try:
            solved = _play_out(book, white, [(7, 7)])
        finally:
            book.close()
    finally:
        white.close()
    return wins, games, len(solved.history), solved.winner


if __name__ == "__main__":
    main()
