"""Tactical puzzles: a position, the side to move, and every correct move.

Each puzzle is a text file under puzzles/<category>/:

    # Blocks an open three with room on one side only.
    to-move: black
    answers: A8 E8 F8

        A B C D E F G H I J K L M N O
      1 . . . . . . . . . . . . . . .
      ...

X is Black, O is White. Squares are named like the web app: column letter,
then row number with row 1 at the top.

    python tools/puzzles.py                 the base at fixed depth 2
    python tools/puzzles.py base:time=1 learned
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.board import BLACK, EMPTY, WHITE, Board, square_name  # noqa: E402
from engine.players import Player, make_player  # noqa: E402

PUZZLES = ROOT / "puzzles"

# In order of difficulty. pytest keeps the quick ones at 100%.
CATEGORIES = (
    "win",
    "block-four",
    "block-three",
    "chain-of-fours",
    "four-three",
    "chain-of-threes",
    "defend-chain",
)
QUICK = CATEGORIES[:5]

_STONES = {".": EMPTY, "X": BLACK, "O": WHITE}
_SIDES = {"black": BLACK, "white": WHITE}


@dataclass
class Puzzle:
    path: Path
    category: str
    note: str
    board: Board
    side: int
    answers: frozenset[tuple[int, int]]

    @property
    def name(self) -> str:
        return f"{self.category}/{self.path.stem}"


def parse_square(name: str) -> tuple[int, int]:
    return int(name[1:]) - 1, ord(name[0].upper()) - ord("A")


def load(path: Path) -> Puzzle:
    note = ""
    side = 0
    answers: set[tuple[int, int]] = set()
    rows: list[list[int]] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            note = note or text.lstrip("# ")
        elif text.startswith("to-move:"):
            side = _SIDES[text.split(":", 1)[1].strip()]
        elif text.startswith("answers:"):
            answers = {parse_square(name) for name in text.split(":", 1)[1].split()}
        elif text[0].isdigit():
            rows.append([_STONES[mark] for mark in text.split()[1:]])
    if not side or not answers or not rows:
        raise ValueError(f"{path}: needs to-move, answers and a board")
    board = Board.from_rows(rows)
    return Puzzle(path, path.parent.name, note, board, side, frozenset(answers))


def format_puzzle(board: Board, side: int, answers: set[tuple[int, int]], note: str) -> str:
    marks = {EMPTY: ".", BLACK: "X", WHITE: "O"}
    lines = [
        f"# {note}",
        f"to-move: {'black' if side == BLACK else 'white'}",
        f"answers: {' '.join(square_name(*square) for square in sorted(answers))}",
        "",
        "     " + " ".join(chr(ord("A") + col) for col in range(board.size)),
    ]
    for row in range(board.size):
        lines.append(f"{row + 1:>4} " + " ".join(marks[board.get(row, col)] for col in range(board.size)))
    return "\n".join(lines) + "\n"


def clear_caches() -> None:
    """Forget earlier answers, so every puzzle is timed and solved from scratch."""
    from engine import tactics, threats

    tactics._CACHE.clear()
    threats._VCT_CACHE.clear()
    threats._VCF_CACHE.clear()


def load_all(categories: tuple[str, ...] = CATEGORIES) -> list[Puzzle]:
    found = []
    for category in categories:
        for path in sorted((PUZZLES / category).glob("*.txt")):
            found.append(load(path))
    return found


def solve(player: Player, puzzle: Puzzle) -> tuple[bool, tuple[int, int] | None, float]:
    clear_caches()
    board = Board(list(puzzle.board.cells), size=puzzle.board.size)
    started = time.perf_counter()
    result = player.move(board, puzzle.side)
    seconds = time.perf_counter() - started
    move = None if result is None else (result.row, result.col)
    return move in puzzle.answers, move, seconds


def run(spec: str, categories: tuple[str, ...], verbose: bool) -> dict[str, tuple[int, int, float]]:
    player = make_player(spec)
    totals: dict[str, tuple[int, int, float]] = {}
    try:
        for puzzle in load_all(categories):
            ok, move, seconds = solve(player, puzzle)
            solved, count, spent = totals.get(puzzle.category, (0, 0, 0.0))
            totals[puzzle.category] = (solved + ok, count + 1, spent + seconds)
            if verbose or not ok:
                played = "nothing" if move is None else square_name(*move)
                wanted = " ".join(square_name(*square) for square in sorted(puzzle.answers))
                status = "ok  " if ok else "MISS"
                print(f"  {status} {puzzle.name}: played {played}, answers {wanted} ({seconds:.2f}s)")
    finally:
        player.close()
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve rate per puzzle category.")
    parser.add_argument("specs", nargs="*", default=["base"], help="bot specs, as in engine/players.py")
    parser.add_argument("--category", action="append", choices=CATEGORIES)
    parser.add_argument("--verbose", action="store_true", help="list solved puzzles too")
    args = parser.parse_args()
    categories = tuple(args.category) if args.category else CATEGORIES
    for spec in args.specs:
        print(spec)
        totals = run(spec, categories, args.verbose)
        for category in categories:
            if category in totals:
                solved, count, spent = totals[category]
                print(f"  {category:<16} {solved:>3}/{count:<3} {solved / count:>5.0%}  {spent / count:.2f}s each")


if __name__ == "__main__":
    main()
