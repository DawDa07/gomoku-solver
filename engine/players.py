"""Every bot behind one call: `move(board, side) -> SearchResult | None`.

`make_player` builds a bot from a short spec, so tools can name bots on the
command line:

    base                      search at fixed depth 2, the same move every time
    base:depth=3              search at fixed depth 3
    base:time=1               search on a one-second clock, like the web app
    learned                   models/best.pt with the threat checks
    learned:path=models/white.pt
    random                    any move near the stones
    book                      the solved rv2 line: the move book, then its threat proof
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

from engine.board import Board
from engine.search import SearchResult, best_move

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPTH = 2


class Player(Protocol):
    name: str

    def move(self, board: Board, side: int) -> SearchResult | None: ...

    def close(self) -> None: ...


class BasePlayer:
    def __init__(self, time_limit: float | None = None, max_depth: int = DEFAULT_DEPTH, name: str = "base"):
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.name = name

    def move(self, board: Board, side: int) -> SearchResult | None:
        return best_move(board, side, time_limit=self.time_limit, max_depth=self.max_depth)

    def close(self) -> None:
        pass


class LearnedPlayer:
    def __init__(self, path: Path, name: str = "learned"):
        import torch

        from engine.net import load_checkpoint

        self.net = load_checkpoint(torch.device("cpu"), path)
        if self.net is None:
            raise FileNotFoundError(f"no checkpoint at {path}")
        self.name = name

    def move(self, board: Board, side: int) -> SearchResult | None:
        from engine.learned import select_action

        if board.winner or not board.legal_moves():
            return None
        row, col, reason, value = select_action(self.net, board, side, temperature=0.0)
        return SearchResult(row, col, int(round(value * 100)), [(row, col)], reason)

    def close(self) -> None:
        pass


class RandomPlayer:
    def __init__(self, seed: int = 0, name: str = "random"):
        self.rng = random.Random(seed)
        self.name = name

    def move(self, board: Board, side: int) -> SearchResult | None:
        moves = board.legal_moves()
        if board.winner or not moves:
            return None
        row, col = self.rng.choice(moves)
        return SearchResult(row, col, 0, [(row, col)], "Random move.")

    def close(self) -> None:
        pass


class BookPlayer:
    """Black's solved strategy: the rv2 book, then the threat proof when a node has no move."""

    def __init__(self, name: str = "book"):
        self.name = name

    def move(self, board: Board, side: int) -> SearchResult | None:
        from engine.book import solved_move

        return solved_move(board, side)

    def close(self) -> None:
        from engine.book import close_solver

        close_solver()


def make_player(spec: str) -> Player:
    kind, _, rest = spec.partition(":")
    options: dict[str, str] = {}
    for part in filter(None, rest.split(",")):
        key, _, value = part.partition("=")
        options[key.strip()] = value.strip()

    def time_option() -> float | None:
        value = options.get("time")
        return None if value in (None, "", "none") else float(value)

    depth = int(options.get("depth", DEFAULT_DEPTH))
    if kind == "base":
        return BasePlayer(time_option(), depth, name=spec)
    if kind == "learned":
        return LearnedPlayer(ROOT / options.get("path", "models/best.pt"), name=spec)
    if kind == "random":
        return RandomPlayer(int(options.get("seed", 0)), name=spec)
    if kind == "book":
        return BookPlayer(name=spec)
    raise ValueError(f"unknown player {spec!r}")
