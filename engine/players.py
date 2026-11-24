"""Every bot behind one call: `move(board, side) -> SearchResult | None`.

`make_player` builds a bot from a short spec, so tools can name bots on the
command line:

    base                      search at fixed depth 2, the same move every time
    base:depth=3              search at fixed depth 3
    base:time=1               search on a one-second clock, like the web app
    learned                   models/learned.pt with the threat checks
    learned:path=models/best.pt
    mcts:sims=200             network-guided tree search, models/best.pt
    random                    any move near the stones
    book                      the solved rv2 line: the move book, then its threat proof
    git:rev=44622f0           the base bot from an older commit, in a subprocess
    dir:path=/tmp/engine-copy the base bot from a copied `engine/` folder
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from engine.board import Board
from engine.search import SearchResult, best_move

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPTH = 2

# Runs inside the subprocess, with the other engine first on sys.path.
_SERVER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from engine.board import Board
from engine.search import best_move
for line in sys.stdin:
    request = json.loads(line)
    board = Board(request["cells"])
    for row, col, player in request["history"]:
        board.history.append((row, col, player))
        board._winner_before.append(0)
    try:
        result = best_move(board, request["side"], time_limit=request["time"], max_depth=request["depth"])
    except TypeError:
        result = best_move(board, request["side"], time_limit=request["time"])
    print(json.dumps(None if result is None else [result.row, result.col]), flush=True)
"""


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


class ExternalPlayer:
    """The base bot from another copy of `engine/`, run in its own process.

    Older engines only know 15×15 boards.
    """

    def __init__(self, root: Path, time_limit: float | None, max_depth: int, name: str):
        self.root = Path(root)
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.name = name
        self.process = subprocess.Popen(
            [sys.executable, "-c", _SERVER, str(self.root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(self.root),
        )

    def move(self, board: Board, side: int) -> SearchResult | None:
        if board.size != 15:
            raise ValueError(f"{self.name} only plays 15×15")
        request = {
            "cells": board.cells,
            "history": board.history,
            "side": side,
            # Older engines have no fixed-depth mode; a huge clock lets every depth finish.
            "time": 1e9 if self.time_limit is None else self.time_limit,
            "depth": self.max_depth,
        }
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        answer = json.loads(self.process.stdout.readline())
        if answer is None:
            return None
        row, col = answer
        return SearchResult(row, col, 0, [(row, col)], f"{self.name} move.")

    def close(self) -> None:
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            self.process.wait(timeout=10)


def checkout(rev: str) -> Path:
    """Extract `engine/` at a git revision into a temporary folder."""
    folder = Path(tempfile.mkdtemp(prefix=f"gomoku-{rev}-"))
    archive = subprocess.run(
        ["git", "archive", rev, "engine"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(folder)], input=archive.stdout, check=True)
    return folder


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
        return LearnedPlayer(ROOT / options.get("path", "models/learned.pt"), name=spec)
    if kind == "random":
        return RandomPlayer(int(options.get("seed", 0)), name=spec)
    if kind == "book":
        return BookPlayer(name=spec)
    if kind == "git":
        return ExternalPlayer(checkout(options["rev"]), time_option(), depth, name=spec)
    if kind == "dir":
        return ExternalPlayer(Path(options["path"]), time_option(), depth, name=spec)
    if kind == "mcts":
        from engine.mcts import MCTSPlayer

        return MCTSPlayer(
            ROOT / options.get("path", "models/best.pt"),
            simulations=int(options.get("sims", 200)),
            name=spec,
        )
    raise ValueError(f"unknown player {spec!r}")
