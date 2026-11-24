"""Threat tactics that decide a move before any search.

In priority order, for the side to move:

1. Make five.
2. Block the opponent's five.
3. Make an open four (two win squares at once).
4. Win by continuous fours.
5. Make a double threat that one reply cannot stop, directly or after a four.
6. Win by a chain of threes and fours.
7. Stop the opponent's open three, broken three, or forced fours, preferring
   blocks that leave them no double threat and no chain of threes.
8. Take away the opponent's double threat or chain of threes before they
   start it.

A tactic names one move, or several equally safe moves for the search or the
network to choose between.
"""

from __future__ import annotations

import time
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
    vct,
    windows,
    wins_after,
)

_CACHE: dict[tuple, Tactic | None] = {}
_CACHE_LIMIT = 50_000
# Replies checked against the opponent's chain of threes, nearest first.
CHAIN_DEFENSE_LIMIT = 16
# Chain-of-threes budgets for checking a reply: a quick screen, then a deeper look.
CHECK_BUDGETS = (40, 250)


@dataclass(frozen=True)
class Tactic:
    moves: tuple[tuple[int, int], ...]
    reason: str
    score: int = 0
    line: tuple[tuple[int, int], ...] = ()


def find_tactic(
    board: Board,
    side: int,
    deadline: float | None = None,
    defense_deadline: float | None = None,
) -> Tactic | None:
    """The tactic for `side` to move, or None in a quiet position.

    `deadline` (a `time.perf_counter()` value) cuts the chain-of-threes search
    short. Stopping the opponent's chain may run on to `defense_deadline`,
    since a missed defense loses the game. Without deadlines, the searches
    stop only on their budgets, so the answer is the same every time.
    """
    key = (board.size, side, tuple(board.cells))
    if key in _CACHE:
        return _CACHE[key]
    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.clear()
    found = _find(board, side, deadline, defense_deadline or deadline)
    _CACHE[key] = found
    return found


def _find(board: Board, side: int, deadline: float | None, defense_deadline: float | None) -> Tactic | None:
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

    if _has_time(deadline):
        line = vct(board, side, deadline)
        if line:
            squares = tuple(at(i) for i in line)
            text = ", ".join(name(i) for i in line)
            return Tactic(
                (squares[0],),
                f"Forced win by threes and fours: {text}.",
                WIN_SCORE - 5,
                squares,
            )

    if threatened(board, other):
        three = bool(open_four_moves(board, other))
        kind = "Blocks the open three." if three else "Stops a forced sequence of fours."
        good = defenses(board, other)
        if good:
            safe = _leave_no_attack(board, side, good)
            if len(safe) > 1:
                safe = _leave_no_chain(board, side, safe, defense_deadline) or safe
            return Tactic(tuple(at(i) for i in safe), kind)
        fallback = sorted(open_four_moves(board, other)) or sorted(scan(board, other)[1])
        return Tactic(
            tuple(at(i) for i in fallback),
            "No single move stops every threat. " + kind,
            -WIN_SCORE // 2,
        )

    attacks = attack_squares(board, other)
    their_line = vct(board, other, defense_deadline) if _has_time(defense_deadline) else None
    if attacks or their_line:
        candidates = attack_defenses(board, other, attacks) if attacks else []
        if their_line:
            candidates += [i for i in _chain_candidates(board, other, their_line) if i not in candidates]
        good = _leave_no_chain(board, side, candidates[:CHAIN_DEFENSE_LIMIT], defense_deadline)
        reason = "Takes away the opponent's double threat." if attacks else "Stops the opponent's chain of threes."
        return Tactic(tuple(at(i) for i in (good or candidates or attacks)), reason)

    return None


def _has_time(deadline: float | None) -> bool:
    return deadline is None or time.perf_counter() < deadline


def _chain_candidates(board: Board, attacker: int, line: list[int]) -> list[int]:
    """Squares on the attacker's winning line, then in the windows of their threats, nearest first."""
    size = board.size
    cells = board.cells
    defender = opponent(attacker)
    found = [i for i in line if cells[i] == EMPTY]
    near: set[int] = set()
    for move in line[::2]:
        for window in windows(size)[1][move]:
            if defender in (cells[i] for i in window):
                continue
            if sum(cells[i] == attacker for i in window) >= 2:
                near.update(i for i in window if cells[i] == EMPTY)
    first = divmod(line[0], size)

    def distance(index: int) -> tuple[int, int]:
        row, col = divmod(index, size)
        return max(abs(row - first[0]), abs(col - first[1])), index

    found += sorted(near - set(found), key=distance)
    return found


def _leave_no_chain(board: Board, side: int, moves: list[int], deadline: float | None) -> list[int]:
    """The moves after which the opponent has no threat and no chain of threes.

    A cheap search screens every move, and the survivors are searched again
    more deeply. Stops at the deadline and returns what has been confirmed.
    """
    safe = moves
    for budget in CHECK_BUDGETS:
        survivors = _no_chain(board, side, safe, deadline, budget)
        if not survivors:
            return [] if safe is moves else safe
        safe = survivors
        if len(safe) == 1:
            break
    return safe


def _no_chain(board: Board, side: int, moves: list[int], deadline: float | None, budget: int) -> list[int]:
    other = opponent(side)
    cells = board.cells
    safe = []
    for move in moves:
        if not _has_time(deadline):
            break
        cells[move] = side
        try:
            if threatened(board, other):
                continue
            chain = vct(board, other, deadline, budget=budget)
            if chain is None and _has_time(deadline):
                safe.append(move)
        finally:
            cells[move] = EMPTY
    return safe


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
