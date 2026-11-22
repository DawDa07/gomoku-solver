"""Threats found on five-cell windows.

A window is five cells in a row along one of the four directions. Every five
fills some window, so counting one player's stones in each window finds solid
and broken shapes alike: XXX.., XX.X., X.X.X.

For one player:

- A win square completes five.
- A four move makes at least one win square.
- An open-four move makes two or more win squares. One reply cannot block
  both, so the player wins next turn. Having such a move available is what an
  open three, solid (.XXX.) or broken (.XX.X.), means.
- VCF, victory by continuous fours, is a chain of four moves where every reply
  is forced, ending in a move that makes two win squares at once.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from engine.board import DIRECTIONS, EMPTY, Board, opponent

VCF_DEPTH = 8
VCF_BUDGET = 300
FORK_DEFENSE_LIMIT = 12


@lru_cache(maxsize=None)
def windows(size: int) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[int, ...], ...], ...]]:
    """Every five-cell window as cell indices, and the windows through each cell."""
    lines: list[tuple[int, ...]] = []
    for row in range(size):
        for col in range(size):
            for dr, dc in DIRECTIONS:
                end_row, end_col = row + 4 * dr, col + 4 * dc
                if 0 <= end_row < size and 0 <= end_col < size:
                    lines.append(tuple((row + i * dr) * size + col + i * dc for i in range(5)))
    through: list[list[tuple[int, ...]]] = [[] for _ in range(size * size)]
    for line in lines:
        for index in line:
            through[index].append(line)
    return tuple(lines), tuple(tuple(found) for found in through)


@lru_cache(maxsize=None)
def window_index(size: int) -> np.ndarray:
    return np.array(windows(size)[0], dtype=np.intp)


def _windows_with(board: Board, player: int, stones: int) -> np.ndarray:
    """Empty cells of windows holding exactly `stones` of `player` and nothing else."""
    index = window_index(board.size)
    values = np.asarray(board.cells, dtype=np.int8)[index]
    empty = values == EMPTY
    own = (values == player).sum(axis=1)
    rows = (own == stones) & (own + empty.sum(axis=1) == 5)
    return index[rows][empty[rows]]


def scan(board: Board, player: int) -> tuple[set[int], set[int]]:
    """Win squares and four moves for `player`, as cell indices."""
    index = window_index(board.size)
    values = np.asarray(board.cells, dtype=np.int8)[index]
    empty = values == EMPTY
    own = (values == player).sum(axis=1)
    free = own + empty.sum(axis=1) == 5
    wins = free & (own == 4)
    fours = free & (own == 3)
    return (
        set(index[wins][empty[wins]].tolist()),
        set(index[fours][empty[fours]].tolist()),
    )


def three_squares(board: Board, player: int) -> set[int]:
    """Moves that put a third stone in some open window for `player`."""
    return set(_windows_with(board, player, 2).tolist())


def wins_after(board: Board, player: int, move: int, wins: set[int]) -> set[int]:
    """Win squares `player` would have after playing `move`."""
    cells = board.cells
    found = set(wins)
    found.discard(move)
    cells[move] = player
    try:
        for line in windows(board.size)[1][move]:
            own = 0
            empty = -1
            for index in line:
                value = cells[index]
                if value == player:
                    own += 1
                elif value == EMPTY:
                    empty = index
                else:
                    break
            else:
                if own == 4:
                    found.add(empty)
    finally:
        cells[move] = EMPTY
    return found


def open_four_moves(
    board: Board,
    player: int,
    wins: set[int] | None = None,
    fours: set[int] | None = None,
) -> set[int]:
    if wins is None or fours is None:
        wins, fours = scan(board, player)
    return {move for move in fours if len(wins_after(board, player, move, wins)) >= 2}


def vcf(board: Board, player: int, depth: int = VCF_DEPTH, budget: int = VCF_BUDGET) -> list[int] | None:
    """A forced win by fours for `player`, as if `player` were to move.

    Returns the line, alternating the attacker's four and the forced block, or
    None. It gives up when the opponent already has a win square, because
    then the opponent's four comes first.
    """
    other = opponent(player)
    if scan(board, other)[0]:
        return None
    return _vcf(board, player, other, depth, [budget])


def _vcf(board: Board, player: int, other: int, depth: int, budget: list[int]) -> list[int] | None:
    wins, fours = scan(board, player)
    if wins:
        return [min(wins)]
    if depth == 0:
        return None
    ranked = []
    for move in fours:
        made = wins_after(board, player, move, wins)
        if made:
            ranked.append((-len(made), move, made))
    ranked.sort(key=lambda item: (item[0], item[1]))
    cells = board.cells
    for count, move, made in ranked:
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        if -count >= 2:
            return [move]
        (block,) = made
        cells[move] = player
        cells[block] = other
        try:
            if not scan(board, other)[0]:
                rest = _vcf(board, player, other, depth - 1, budget)
                if rest is not None:
                    return [move, block, *rest]
        finally:
            cells[move] = EMPTY
            cells[block] = EMPTY
    return None


def threatened(board: Board, attacker: int) -> bool:
    """Whether `attacker`, if it were to move, wins by a four or an open four."""
    wins, fours = scan(board, attacker)
    if wins:
        return True
    if open_four_moves(board, attacker, wins, fours):
        return True
    return vcf(board, attacker) is not None


def defenses(board: Board, attacker: int, depth: int = 1, first: bool = False) -> list[int]:
    """Replies that leave `attacker` with no win square, open four, or VCF.

    Besides blocking directly, the defender may play a four of its own. The
    attacker must answer it, and the defender then needs a defense in the new
    position. `depth` bounds how many such counter-fours are chained.
    """
    defender = opponent(attacker)
    wins, fours = scan(board, attacker)
    cells = board.cells
    good: list[int] = []
    blocks = wins if wins else wins | fours
    for square in sorted(blocks):
        cells[square] = defender
        try:
            if not threatened(board, attacker):
                good.append(square)
                if first:
                    return good
        finally:
            cells[square] = EMPTY
    if wins or depth <= 0:
        return good
    own_wins, own_fours = scan(board, defender)
    if own_wins:
        good.append(min(own_wins))
        return good
    for square in sorted(own_fours - blocks):
        made = wins_after(board, defender, square, own_wins)
        if not made:
            continue
        if len(made) >= 2:
            good.append(square)
            if first:
                return good
            continue
        (reply,) = made
        cells[square] = defender
        cells[reply] = attacker
        try:
            if can_defend(board, attacker, depth - 1):
                good.append(square)
                if first:
                    return good
        finally:
            cells[square] = EMPTY
            cells[reply] = EMPTY
    return good


def can_defend(board: Board, attacker: int, depth: int = 1) -> bool:
    """Whether the defender, to move, can stop every threat `attacker` has."""
    if not threatened(board, attacker):
        return True
    if vcf(board, opponent(attacker)) is not None:
        return True
    return bool(defenses(board, attacker, depth, first=True))


def fork_moves(board: Board, player: int) -> list[int]:
    """Quiet moves that leave threats no reply can stop, counter-fours included."""
    cells = board.cells
    found: list[int] = []
    for move in sorted(three_squares(board, player)):
        cells[move] = player
        try:
            wins, fours = scan(board, player)
            if wins:
                continue
            if len(open_four_moves(board, player, wins, fours)) < 2:
                continue
            if not can_defend(board, player):
                found.append(move)
        finally:
            cells[move] = EMPTY
    return found


def four_into_fork(board: Board, attacker: int) -> int | None:
    """A four whose forced block leaves `attacker` a fork. Returns the four."""
    defender = opponent(attacker)
    wins, fours = scan(board, attacker)
    if wins:
        return None
    cells = board.cells
    for move in sorted(fours):
        made = wins_after(board, attacker, move, wins)
        if len(made) != 1:
            continue
        (reply,) = made
        cells[move] = attacker
        cells[reply] = defender
        try:
            if scan(board, defender)[0]:
                continue
            if fork_moves(board, attacker):
                return move
        finally:
            cells[move] = EMPTY
            cells[reply] = EMPTY
    return None


def attack_squares(board: Board, attacker: int) -> list[int]:
    """Fork moves, or a four that leads into one, for `attacker`."""
    forks = fork_moves(board, attacker)
    if forks:
        return forks
    four = four_into_fork(board, attacker)
    return [] if four is None else [four]


def attack_defenses(board: Board, attacker: int, squares: list[int]) -> list[int]:
    """Replies after which `attacker` has no fork and no four leading into one."""
    defender = opponent(attacker)
    size = board.size
    cells = board.cells
    candidates = set(squares)
    for square in squares:
        for line in windows(size)[1][square]:
            if any(cells[i] == defender for i in line):
                continue
            candidates.update(i for i in line if cells[i] == EMPTY)

    def distance(index: int) -> int:
        row, col = divmod(index, size)
        return min(max(abs(row - sr), abs(col - sc)) for sr, sc in (divmod(s, size) for s in squares))

    good: list[int] = []
    for square in sorted(candidates, key=lambda i: (distance(i), i))[:FORK_DEFENSE_LIMIT]:
        cells[square] = defender
        try:
            if not attack_squares(board, attacker):
                good.append(square)
        finally:
            cells[square] = EMPTY
    return good
