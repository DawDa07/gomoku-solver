"""Mine puzzles from tournament games saved in results/.

    python tools/mine_puzzles.py chain-of-fours chain-of-threes defend-chain

The answers come from the threat searches run with far larger budgets than a
bot gets during a game:

- chain-of-fours: the side to move wins by fours, and has no double threat
  or open four to play instead. Answers: every first four of a winning chain,
  plus any immediate double threat.
- chain-of-threes: the side to move wins by threes and fours, but not by
  fours alone. Answers: every first move of such a chain.
- defend-chain: the opponent has a chain of threes, but no four or open
  three yet. Answers: every move after which they have neither.

Mined files are named mined-NN.txt and replaced on every run. Hand-made
puzzles in the same folders are left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.board import BLACK, EMPTY, Board, opponent  # noqa: E402
from engine.threats import (  # noqa: E402
    fork_moves,
    open_four_moves,
    scan,
    threatened,
    vcf,
    vct,
    vct_moves,
    wins_after,
)
from tools.puzzles import PUZZLES, format_puzzle  # noqa: E402

RESULTS = ROOT / "results"
DEEP_VCF = {"depth": 12, "budget": 5000}
DEEP_VCT = {"depth": 8, "budget": 1500}
MAX_ANSWERS = 4


def positions(limit_games: int | None = None):
    """(board, side to move, label) for every position of every saved game, losses first."""
    seen: set[tuple] = set()
    for path in sorted(RESULTS.glob("*.json")):
        records = json.loads(path.read_text())["records"]
        for number, record in enumerate(records[:limit_games]):
            if record.get("winner", 0) == 0:
                continue
            board = Board()
            side = BLACK
            for ply, (row, col) in enumerate(record["moves"]):
                key = tuple(board.cells)
                if ply >= 8 and key not in seen:
                    seen.add(key)
                    label = f"{record['black']} vs {record['white']}, move {ply + 1}"
                    yield Board(list(board.cells)), side, label, f"{path.name}:{number}"
                board.place(row, col, side)
                side = opponent(side)


def _quiet(board: Board, side: int) -> bool:
    """Neither side has a four, and the side to move has no open four."""
    wins, fours = scan(board, side)
    return not wins and not scan(board, opponent(side))[0] and not open_four_moves(board, side, wins, fours)


def chain_of_fours(board: Board, side: int) -> set[int] | None:
    if not _quiet(board, side) or fork_moves(board, side):
        return None
    line = vcf(board, side, **DEEP_VCF)
    if not line or len(line) < 5:
        return None
    found = set()
    wins, fours = scan(board, side)
    cells = board.cells
    for move in fours:
        made = wins_after(board, side, move, wins)
        if len(made) != 1:
            continue
        (block,) = made
        cells[move] = side
        cells[block] = opponent(side)
        try:
            if not scan(board, opponent(side))[0] and vcf(board, side, **DEEP_VCF):
                found.add(move)
        finally:
            cells[move] = EMPTY
            cells[block] = EMPTY
    return found


def chain_of_threes(board: Board, side: int) -> set[int] | None:
    if not _quiet(board, side) or vcf(board, side, **DEEP_VCF) or fork_moves(board, side):
        return None
    if threatened(board, opponent(side)):
        return None
    if vct(board, side) is None:
        return None
    return set(vct_moves(board, side, **DEEP_VCT))


def defend_chain(board: Board, side: int) -> set[int] | None:
    other = opponent(side)
    if not _quiet(board, side) or threatened(board, other):
        return None
    # The quick search first: a deep search on every quiet position would take hours.
    if vcf(board, side) or vct(board, side) or vct(board, other) is None:
        return None
    good = set()
    cells = board.cells
    for row, col in board.legal_moves():
        move = row * board.size + col
        cells[move] = side
        try:
            if not threatened(board, other) and vct(board, other, **DEEP_VCT) is None:
                good.add(move)
        finally:
            cells[move] = EMPTY
    return good


FINDERS = {
    "chain-of-fours": chain_of_fours,
    "chain-of-threes": chain_of_threes,
    "defend-chain": defend_chain,
}


def _check(args):
    category, cells, side, label, game = args
    board = Board(cells)
    answers = FINDERS[category](board, side)
    return category, cells, side, label, game, answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine puzzles from saved games.")
    parser.add_argument("categories", nargs="+", choices=sorted(FINDERS))
    parser.add_argument("--count", type=int, default=12, help="puzzles per category")
    parser.add_argument("--games", type=int, default=None, help="games per results file")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    tasks = [
        (category, board.cells, side, label, game)
        for board, side, label, game in positions(args.games)
        for category in args.categories
    ]
    for category in args.categories:
        folder = PUZZLES / category
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("mined-*.txt"):
            old.unlink()

    counts = dict.fromkeys(args.categories, 0)
    games_used: dict[str, set] = {category: set() for category in args.categories}
    with get_context("spawn").Pool(args.workers) as pool:
        for category, cells, side, label, game, answers in pool.imap_unordered(_check, tasks, chunksize=4):
            if not answers or len(answers) > MAX_ANSWERS:
                continue
            if counts[category] >= args.count or game in games_used[category]:
                continue
            games_used[category].add(game)
            counts[category] += 1
            board = Board(cells)
            squares = {divmod(index, board.size) for index in answers}
            note = f"Mined from {label}."
            path = PUZZLES / category / f"mined-{counts[category]:02d}.txt"
            path.write_text(format_puzzle(board, side, squares, note))
            print(f"{category} {path.name}: {label}", flush=True)
            if all(counts[c] >= args.count for c in args.categories):
                pool.terminate()
                break
    print(counts)


if __name__ == "__main__":
    main()
