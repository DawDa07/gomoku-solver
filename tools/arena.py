"""Play two bots against each other from fixed openings.

Each opening is three stones near the center (Black, White, Black). It is
played twice, once with each bot as Black, so neither side gets the easier
color more often. Bots are named with the specs in `engine/players.py`.

    python tools/arena.py base learned --openings 50
    python tools/arena.py base:depth=3 dir:path=/tmp/old,depth=2 --workers 6

Results are printed and saved to results/<date>-<a>-vs-<b>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.board import BLACK, WHITE, Board  # noqa: E402
from engine.players import Player, make_player  # noqa: E402

RESULTS = ROOT / "results"


def openings(count: int, size: int = 15, seed: int = 0) -> list[list[tuple[int, int]]]:
    """`count` different three-stone openings within two of the center."""
    rng = random.Random(seed)
    mid = size // 2
    found: list[list[tuple[int, int]]] = []
    seen: set[tuple] = set()
    while len(found) < count:
        stones: list[tuple[int, int]] = []
        while len(stones) < 3:
            square = (mid + rng.randint(-2, 2), mid + rng.randint(-2, 2))
            if square not in stones:
                stones.append(square)
        key = (stones[0], stones[1], frozenset((stones[0], stones[2])))
        if key in seen:
            continue
        seen.add(key)
        found.append(stones)
    return found


@dataclass
class Game:
    opening: int
    black: str
    white: str
    winner: int
    moves: list[tuple[int, int]] = field(default_factory=list)
    seconds: float = 0.0


def play_game(black: Player, white: Player, opening: list[tuple[int, int]], size: int, index: int) -> Game:
    board = Board(size=size)
    side = BLACK
    for row, col in opening:
        board.place(row, col, side)
        side = WHITE if side == BLACK else BLACK
    started = time.perf_counter()
    while not board.winner and not board.is_full():
        player = black if side == BLACK else white
        result = player.move(board, side)
        if result is None:
            break
        board.place(result.row, result.col, side)
        side = WHITE if side == BLACK else BLACK
    return Game(
        opening=index,
        black=black.name,
        white=white.name,
        winner=board.winner,
        moves=[(row, col) for row, col, _ in board.history],
        seconds=time.perf_counter() - started,
    )


_PLAYERS: dict[str, Player] = {}


def _player(spec: str) -> Player:
    if spec not in _PLAYERS:
        _PLAYERS[spec] = make_player(spec)
    return _PLAYERS[spec]


def _close_players() -> None:
    for player in _PLAYERS.values():
        player.close()
    _PLAYERS.clear()


def _task(args: tuple[str, str, list[tuple[int, int]], int, int]) -> Game:
    black, white, opening, size, index = args
    return play_game(_player(black), _player(white), opening, size, index)


def run_match(
    a: str,
    b: str,
    count: int = 50,
    size: int = 15,
    workers: int | None = None,
    seed: int = 0,
    progress: bool = True,
) -> dict:
    """Play every opening twice, swapping colors. Returns the summary dict."""
    tasks = []
    for index, opening in enumerate(openings(count, size, seed)):
        tasks.append((a, b, opening, size, index))
        tasks.append((b, a, opening, size, index))
    workers = workers or max(1, min(len(tasks), (os.cpu_count() or 2) - 1))
    started = time.perf_counter()
    games: list[Game] = []
    if workers == 1:
        for done, task in enumerate(tasks, 1):
            games.append(_task(task))
            if progress:
                _progress(done, len(tasks))
        _close_players()
    else:
        with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            for done, game in enumerate(pool.imap_unordered(_task, tasks), 1):
                games.append(game)
                if progress:
                    _progress(done, len(tasks))
    if progress:
        print(file=sys.stderr)
    games.sort(key=lambda game: (game.opening, game.black != a))
    return summarize(a, b, games, size, time.perf_counter() - started)


def _init_worker() -> None:
    import atexit

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    atexit.register(_close_players)


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} games", end="", file=sys.stderr, flush=True)


def summarize(a: str, b: str, games: list[Game], size: int, seconds: float) -> dict:
    record = {
        "black": {"wins": 0, "losses": 0, "draws": 0},
        "white": {"wins": 0, "losses": 0, "draws": 0},
    }
    loss_lengths: list[int] = []
    win_lengths: list[int] = []
    for game in games:
        color = "black" if game.black == a else "white"
        mine = BLACK if color == "black" else WHITE
        if game.winner == 0:
            record[color]["draws"] += 1
        elif game.winner == mine:
            record[color]["wins"] += 1
            win_lengths.append(len(game.moves))
        else:
            record[color]["losses"] += 1
            loss_lengths.append(len(game.moves))
    total = len(games)
    wins = record["black"]["wins"] + record["white"]["wins"]
    draws = record["black"]["draws"] + record["white"]["draws"]
    score = (wins + 0.5 * draws) / total if total else 0.0
    elo, low, high = rating(score, total)
    return {
        "a": a,
        "b": b,
        "size": size,
        "games": total,
        "score": score,
        "elo": elo,
        "elo_low": low,
        "elo_high": high,
        "a_as": record,
        "mean_length_a_wins": _mean(win_lengths),
        "mean_length_a_losses": _mean(loss_lengths),
        "seconds": seconds,
        "records": [asdict(game) for game in games],
    }


def rating(score: float, games: int) -> tuple[float, float, float]:
    """Rating difference for a score, with a 95% range from the game count."""
    if games == 0:
        return 0.0, 0.0, 0.0
    edge = 0.5 / games
    error = math.sqrt(max(score * (1 - score), edge) / games)

    def elo(fraction: float) -> float:
        fraction = min(max(fraction, edge), 1 - edge)
        return 400 * math.log10(fraction / (1 - fraction))

    return elo(score), elo(score - 1.96 * error), elo(score + 1.96 * error)


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def report(summary: dict) -> str:
    a, b = summary["a"], summary["b"]
    black, white = summary["a_as"]["black"], summary["a_as"]["white"]
    lines = [
        f"{a} vs {b} on {summary['size']}×{summary['size']}, {summary['games']} games "
        f"in {summary['seconds']:.0f}s",
        f"  {a} as Black: {black['wins']} wins, {black['losses']} losses, {black['draws']} draws",
        f"  {a} as White: {white['wins']} wins, {white['losses']} losses, {white['draws']} draws",
        f"  score {summary['score']:.1%}, rating difference {summary['elo']:+.0f} "
        f"(95% range {summary['elo_low']:+.0f} to {summary['elo_high']:+.0f})",
    ]
    if summary["mean_length_a_losses"] is not None:
        lines.append(f"  {a}'s losses last {summary['mean_length_a_losses']:.0f} moves on average")
    return "\n".join(lines)


def save(summary: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)

    def slug(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9.=-]+", "_", text).strip("_")[:60]

    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = RESULTS / f"{stamp}-{slug(summary['a'])}-vs-{slug(summary['b'])}.json"
    path.write_text(json.dumps(summary, indent=1))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Play two bots from fixed openings.")
    parser.add_argument("a", help="first bot spec, for example base or learned")
    parser.add_argument("b", help="second bot spec")
    parser.add_argument("--openings", type=int, default=50)
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run_match(args.a, args.b, args.openings, args.size, args.workers, args.seed)
    print(report(summary))
    print(f"  saved {save(summary).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
