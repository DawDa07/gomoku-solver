"""Threat tactics that decide a move before any search.

In priority order, for the side to move:

1. Make five.
2. Block the opponent's five.
3. Make an open four (two win squares at once).
4. Win by continuous fours.
5. Make a double threat that one reply cannot stop, directly or after a four.
6. Stop the opponent's open three, broken three, or forced fours, preferring
   blocks that do not leave them a double threat.
7. Take away the opponent's double threat, or the four that leads into one,
   before they play it.

A tactic names one move, or several equally safe moves for the search or the
network to choose between.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.board import EMPTY, Board, opponent, square_name
from engine.evaluate import WIN_SCORE
from engine.threats import (
    attack_defenses,
    attack_squares,
    defenses,
    fork_moves,
    four_into_fork,
    open_four_moves,
    scan,
    threatened,
    vcf,
    wins_after,
)

_CACHE: dict[tuple, Tactic | None] = {}
_CACHE_LIMIT = 50_000


@dataclass(frozen=True)
class Tactic:
    moves: tuple[tuple[int, int], ...]
    reason: str
    score: int = 0
    line: tuple[tuple[int, int], ...] = ()


def find_tactic(board: Board, side: int) -> Tactic | None:
    key = (board.size, side, tuple(board.cells))
    if key in _CACHE:
        return _CACHE[key]
    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.clear()
    found = _find(board, side)
    _CACHE[key] = found
    return found


def _find(board: Board, side: int) -> Tactic | None:
    size = board.size
    other = opponent(side)

    def at(index: int) -> tuple[int, int]:
        return divmod(index, size)

    def name(index: int) -> str:
        return square_name(*at(index))

    wins, fours = scan(board, side)
    if wins:
        move = min(wins)
        return Tactic((at(move),), "Takes the win.", WIN_SCORE, (at(move),))

    their_wins, _ = scan(board, other)
    if len(their_wins) == 1:
        (move,) = their_wins
        return Tactic((at(move),), f"Blocks the opponent's four at {name(move)}.")
    if their_wins:
        move = min(their_wins)
        return Tactic(
            (at(move),),
            "The opponent has two ways to make five. Blocking one of them.",
            -WIN_SCORE,
        )

    opens = open_four_moves(board, side, wins, fours)
    if opens:
        move = max(opens, key=lambda i: (len(wins_after(board, side, i, wins)), -i))
        return Tactic(
            (at(move),),
            "Makes an open four. Five comes next move.",
            WIN_SCORE - 1,
            (at(move),),
        )

    line = vcf(board, side)
    if line:
        squares = tuple(at(i) for i in line)
        text = ", ".join(name(i) for i in line)
        return Tactic(
            (squares[0],),
            f"Forced win by continuous fours: {text}.",
            WIN_SCORE - 2,
            squares,
        )

    forks = fork_moves(board, side)
    if forks:
        move = forks[0]
        return Tactic(
            (at(move),),
            "Makes two threats at once. One reply cannot stop both.",
            WIN_SCORE - 3,
            (at(move),),
        )

    four = four_into_fork(board, side)
    if four is not None:
        return Tactic(
            (at(four),),
            "Plays a four. After the forced block, a double threat follows.",
            WIN_SCORE - 4,
            (at(four),),
        )

    if threatened(board, other):
        three = bool(open_four_moves(board, other))
        kind = "Blocks the open three." if three else "Stops a forced sequence of fours."
        good = defenses(board, other)
        if good:
            return Tactic(tuple(at(i) for i in _leave_no_attack(board, side, good)), kind)
        fallback = sorted(open_four_moves(board, other)) or sorted(scan(board, other)[1])
        return Tactic(
            tuple(at(i) for i in fallback),
            "No single move stops every threat. " + kind,
            -WIN_SCORE // 2,
        )

    attacks = attack_squares(board, other)
    if attacks:
        good = attack_defenses(board, other, attacks)
        return Tactic(
            tuple(at(i) for i in (good or attacks)),
            "Takes away the opponent's double threat.",
        )

    return None


def _leave_no_attack(board: Board, side: int, moves: list[int]) -> list[int]:
    """The moves after which the opponent has no double threat to play. All moves if none do."""
    other = opponent(side)
    cells = board.cells
    safe = []
    for move in moves:
        cells[move] = side
        try:
            if not attack_squares(board, other):
                safe.append(move)
        finally:
            cells[move] = EMPTY
    return safe or moves
