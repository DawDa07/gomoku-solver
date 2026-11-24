"""The solved Black strategy from the rv2 move book.

`gomoku.json` stores one winning line for Black and, at White's turns, the
replies annotated with how many moves that win still takes. The largest count
is the longest delay. Coordinates in the file put row 15 at the top; this
board puts row 1 at the top. Both call the center h8.

The tree is stored in one orientation. White's first move picks which of the
eight symmetries later moves use, the same rule as `coordTransCode` and
`Layout.transSquare` in the rv2 sources. When a node is only a distance and
not a list of moves, `vendor/rv2/solve.js` runs that proof.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from engine.board import BLACK, WHITE, Board, square_name
from engine.search import SearchResult

ROOT = Path(__file__).resolve().parents[1]
BOOK_PATH = ROOT / "vendor" / "rv2" / "gomoku.json"
SOLVER_PATH = ROOT / "vendor" / "rv2" / "solve.js"
CACHE_PATH = ROOT / "results" / "book-cache.json"

_BOOK: dict | None = None
_COUNTS: dict[int, int] = {}
_SOLVER: Solver | None = None


def book_key(row: int, col: int) -> str:
    """rv2 coordinate for one of our squares. Row 0 on our board is their row 15."""
    return f"{chr(ord('a') + col)}{15 - row}"


def parse_book_key(key: str) -> tuple[int, int]:
    """Our row and column for an rv2 coordinate such as h8 or h9."""
    col = ord(key[0]) - ord("a")
    their_row = int(key[1:])
    return 15 - their_row, col


def _square(row: int, col: int) -> int:
    return (row << 4) | col


def _row_col(square: int) -> tuple[int, int]:
    return square >> 4, square & 0xF


def square_trans_code(square: int) -> int:
    """Which of the eight symmetries folds this square into the stored octant."""
    offset = square & 0xF
    number = square >> 4
    if number <= 7:
        if offset <= 7:
            return 7 if offset <= number else 6
        return 5 if (14 - offset) <= number else 0
    if offset <= 7:
        return 1 if offset <= (14 - number) else 4
    return 3 if (14 - offset) <= (14 - number) else 2


def trans_square(square: int, trans_code: int, back: bool) -> int:
    """Rotate or mirror a square. `back` maps a board square into the book."""
    row, col = _row_col(square)
    if trans_code == 1:
        return _square(col, row)
    if trans_code == 2:
        return _square(14 - row, col)
    if trans_code == 3:
        if back:
            return _square(14 - col, row)
        return _square(col, 14 - row)
    if trans_code == 4:
        return _square(14 - row, 14 - col)
    if trans_code == 5:
        return _square(14 - col, 14 - row)
    if trans_code == 6:
        return _square(row, 14 - col)
    if trans_code == 7:
        if back:
            return _square(col, 14 - row)
        return _square(14 - col, row)
    return square


def load_book() -> dict:
    global _BOOK
    if _BOOK is None:
        _BOOK = json.loads(BOOK_PATH.read_text())
    return _BOOK


def node_count(node: dict | int) -> int:
    """Moves-left label stored on a node. A branch's label is one more than its deepest child."""
    if isinstance(node, int):
        return node
    cached = _COUNTS.get(id(node))
    if cached is not None:
        return cached
    best = 0
    for key, child in node.items():
        if key == "count":
            continue
        best = max(best, node_count(child))
    _COUNTS[id(node)] = best + 1
    return best + 1


def _children(node: dict | int) -> list[tuple[str, dict | int]]:
    if not isinstance(node, dict):
        return []
    return [(key, child) for key, child in node.items() if key != "count"]


def _locate(history: list[tuple[int, int, int]]) -> tuple[str, dict | int | None, int]:
    """Walk the played moves. Returns kind, the node reached, and the symmetry code.

    kind is "move" when that node lists next moves, "leaf" when the book knows
    the position but only stored a distance, and "miss" when the sequence leaves the tree.
    """
    node: dict | int = load_book()
    trans = 0
    for index, (row, col, player) in enumerate(history):
        expected = BLACK if index % 2 == 0 else WHITE
        if player != expected or not isinstance(node, dict):
            return "miss", None, trans
        square = _square(row, col)
        if index == 1:
            trans = square_trans_code(square)
        square = trans_square(square, trans, True)
        key = book_key(*_row_col(square))
        if key not in node:
            return "miss", None, trans
        node = node[key]
    if isinstance(node, dict) and _children(node):
        return "move", node, trans
    if isinstance(node, (dict, int)):
        return "leaf", node, trans
    return "miss", None, trans


@dataclass(frozen=True)
class Lookup:
    kind: str
    move: SearchResult | None = None


def lookup(board: Board, side: int) -> Lookup:
    """The stored move for `side` to move, when this sequence is an explicit book node."""
    if board.size != 15:
        return Lookup("miss")
    history = list(board.history)
    expected = BLACK if len(history) % 2 == 0 else WHITE
    if side != expected:
        return Lookup("miss")
    kind, node, trans = _locate(history)
    if kind != "move" or not isinstance(node, dict):
        return Lookup(kind)
    counted = [(key, node_count(child)) for key, child in _children(node)]
    best = max(count for _, count in counted)
    key = sorted(name for name, count in counted if count == best)[0]
    row, col = _row_col(trans_square(_square(*parse_book_key(key)), trans, False))
    if side == BLACK:
        reason = f"Book line. Black's winning move, with {best} moves left in the stored line."
    else:
        reason = f"Book line. White's longest delay, with {best} moves left in the stored line."
    move = SearchResult(row, col, best, [(row, col)], reason)
    return Lookup("move", move)


def stored_replies(board: Board) -> list[tuple[int, int, int]] | None:
    """Every stored reply and its count, or None when this sequence is not an explicit node."""
    if board.size != 15:
        return None
    kind, node, trans = _locate(list(board.history))
    if kind != "move" or not isinstance(node, dict):
        return None
    replies = []
    for key, child in _children(node):
        row, col = _row_col(trans_square(_square(*parse_book_key(key)), trans, False))
        replies.append((row, col, node_count(child)))
    return replies


def iter_white_targets():
    """Each White-to-move book node, as played moves and the replies with the largest count."""
    board = Board()

    def walk(node: dict, ply: int):
        kids = _children(node)
        if ply % 2 == 1 and kids:
            counted = [(key, node_count(child)) for key, child in kids]
            best = max(count for _, count in counted)
            targets = [parse_book_key(key) for key, count in counted if count == best]
            yield list(board.history), targets
        side = BLACK if ply % 2 == 0 else WHITE
        for key, child in kids:
            if not isinstance(child, dict):
                continue
            row, col = parse_book_key(key)
            board.place(row, col, side)
            yield from walk(child, ply + 1)
            board.undo()

    yield from walk(load_book(), 0)


class Solver:
    """Long-running Node process for positions the JSON does not list move by move."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.cache: dict[str, dict | None] = {}
        if CACHE_PATH.is_file():
            try:
                self.cache = json.loads(CACHE_PATH.read_text())
            except json.JSONDecodeError:
                self.cache = {}

    def ask(self, moves: list[tuple[int, int]]) -> dict | None:
        key = ";".join(f"{row},{col}" for row, col in moves)
        if key in self.cache:
            return self.cache[key]
        self._start()
        if self.process is None or self.process.poll() is not None:
            return None
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps({"moves": moves}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            self.close()
            return None
        answer = json.loads(line)
        stored = None if answer.get("row") is None else answer
        self.cache[key] = stored
        self._save()
        return stored

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        node = shutil.which("node")
        if node is None or not SOLVER_PATH.is_file():
            return
        self.process = subprocess.Popen(
            [node, str(SOLVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SOLVER_PATH.parent),
        )

    def _save(self) -> None:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        CACHE_PATH.write_text(json.dumps(self.cache))

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=10)


def _solver() -> Solver:
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = Solver()
    return _SOLVER


def solver_running() -> bool:
    return _SOLVER is not None and _SOLVER.process is not None and _SOLVER.process.poll() is None


def close_solver() -> None:
    global _SOLVER
    if _SOLVER is not None:
        _SOLVER.close()
        _SOLVER = None


def prove(board: Board, side: int) -> SearchResult | None:
    """One move from the threat proof, or None when it finds nothing to play."""
    if board.size != 15 or board.winner:
        return None
    moves = [(row, col) for row, col, _player in board.history]
    answer = _solver().ask(moves)
    if answer is None:
        return None
    state = int(answer["state"])
    # A positive state is a win for the side to move. White keeps a negative
    # state too: that is a proved loss, and the move is one that delays it.
    if side == BLACK and state <= 0:
        return None
    if side == WHITE and state == 0:
        return None
    row, col = int(answer["row"]), int(answer["col"])
    if not board.in_bounds(row, col) or board.get(row, col) != 0:
        return None
    if state > 0:
        reason = f"Solved line. The threat proof wins with {square_name(row, col)}."
    else:
        reason = f"Solved line. {square_name(row, col)} is a delaying reply in the threat proof."
    return SearchResult(row, col, abs(state), [(row, col)], reason)


def solved_move(board: Board, side: int) -> SearchResult | None:
    """Book move when one is stored, otherwise the threat proof."""
    found = lookup(board, side)
    if found.move is not None:
        return found.move
    if board.winner or not board.legal_moves():
        return None
    return prove(board, side)
