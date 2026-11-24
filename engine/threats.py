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

import time
from functools import lru_cache

import numpy as np

from engine.board import DIRECTIONS, EMPTY, Board, opponent

VCF_DEPTH = 8
VCF_BUDGET = 300
FORK_DEFENSE_LIMIT = 12
_VCF_CACHE: dict[tuple, tuple[int, ...] | None] = {}
_VCF_CACHE_LIMIT = 200_000
VCT_DEPTH = 6
VCT_BUDGET = 100
VCT_WIDTH = 12
VCT_VCF_BUDGET = 60
# Counter-fours the defender may chain against one three before the search stops looking.
VCT_COUNTERS = 3


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


# For each player, a cell's code: 1 for their stone, 8 for the opponent's, 0 if
# empty. A window summing to 4 holds four of theirs and one empty cell.
_CODES = np.array([[0, 0, 0], [0, 1, 8], [0, 8, 1]], dtype=np.int8)


def scan(board: Board, player: int) -> tuple[set[int], set[int]]:
    """Win squares and four moves for `player`, as cell indices."""
    index = window_index(board.size)
    values = _CODES[player][np.asarray(board.cells)][index]
    sums = values.sum(axis=1)
    wins = sums == 4
    fours = sums == 3
    return (
        set(index[wins][values[wins] == EMPTY].tolist()),
        set(index[fours][values[fours] == EMPTY].tolist()),
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
    key = (board.size, player, depth, budget, tuple(board.cells))
    if key in _VCF_CACHE:
        line = _VCF_CACHE[key]
        return None if line is None else list(line)
    if len(_VCF_CACHE) > _VCF_CACHE_LIMIT:
        _VCF_CACHE.clear()
    other = opponent(player)
    line = None if scan(board, other)[0] else _vcf(board, player, other, depth, [budget])
    _VCF_CACHE[key] = None if line is None else tuple(line)
    return line


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


class _OutOfTime(Exception):
    pass


_VCT_CACHE: dict[tuple, list[int] | None] = {}
_VCT_CACHE_LIMIT = 200_000


def vct(
    board: Board,
    attacker: int,
    deadline: float | None = None,
    budget: int = VCT_BUDGET,
    depth: int = VCT_DEPTH,
) -> list[int] | None:
    """A forced win by threes and fours for `attacker`, as if `attacker` were to move.

    Each attacking move is a four, which has one reply, or a move that makes
    an open three. The defender answers a three with every block that stops
    the open four, and with every counter-four. The attacker wins on a chain
    of fours or an open four. Returns the line (attacker's move, reply,
    attacker's move, ...) for the first reply, or None when no win is found
    within `depth` attacking moves, `budget` attacking moves tried, and the
    `deadline` (a `time.perf_counter()` value).
    """
    if len(_VCT_CACHE) > _VCT_CACHE_LIMIT:
        _VCT_CACHE.clear()
    search = _ChainSearch(board, attacker, deadline, budget)
    try:
        # Shallow first: a deep search can spend the whole budget before
        # reaching a short win.
        for limit in range(1, depth + 1):
            line = search.attack(limit)
            if line is not None or search.budget[0] <= 0:
                return line
    except _OutOfTime:
        pass
    return None


def vct_moves(
    board: Board,
    attacker: int,
    deadline: float | None = None,
    budget: int = VCT_BUDGET,
    depth: int = VCT_DEPTH,
) -> list[int]:
    """Every first move that starts a chain of threes or fours, each searched on its own budget."""
    search = _ChainSearch(board, attacker, deadline, budget)
    wins, _ = scan(board, attacker)
    if wins:
        return sorted(wins)
    if scan(board, search.defender)[0]:
        return []
    found = []
    try:
        for move, kind in search.candidates():
            search.budget = [budget]
            for limit in range(1, depth + 1):
                if search.try_move(move, kind, limit) is not None:
                    found.append(move)
                    break
                if search.budget[0] <= 0:
                    break
    except _OutOfTime:
        pass
    return found


class _ChainSearch:
    def __init__(self, board: Board, attacker: int, deadline: float | None, budget: int):
        self.board = board
        self.attacker = attacker
        self.defender = opponent(attacker)
        self.deadline = deadline
        self.budget = [budget]
        self.defended: dict[tuple, list[int] | None] = {}

    def _tick(self) -> bool:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise _OutOfTime
        if self.budget[0] <= 0:
            return False
        self.budget[0] -= 1
        return True

    def attack(self, depth: int) -> list[int] | None:
        """Attacker to move: a winning line, or None."""
        board = self.board
        key = (board.size, self.attacker, depth, tuple(board.cells))
        if key in _VCT_CACHE:
            return _VCT_CACHE[key]
        wins, fours = scan(board, self.attacker)
        if wins:
            return [min(wins)]
        if scan(board, self.defender)[0]:
            return None
        opens = open_four_moves(board, self.attacker, wins, fours)
        if opens:
            return [min(opens)]
        line = vcf(board, self.attacker, budget=VCT_VCF_BUDGET)
        if line:
            return line
        if depth <= 0:
            return None
        found = None
        for move, kind in self.candidates(fours):
            if not self._tick():
                break
            found = self.try_move(move, kind, depth)
            if found is not None:
                break
        # A miss after the budget ran out may be wrong, so only certain answers are kept.
        if found is not None or self.budget[0] > 0:
            _VCT_CACHE[key] = found
        return found

    def candidates(self, fours: set[int] | None = None) -> list[tuple[int, str]]:
        """Fours, then moves that make an open three, most promising first."""
        board = self.board
        if fours is None:
            fours = scan(board, self.attacker)[1]
        found = [(move, "four") for move in sorted(fours) if wins_after(board, self.attacker, move, set())]
        threes = []
        for move in three_squares(board, self.attacker) - fours:
            count = len(_open_fours_made(board, self.attacker, move))
            if count:
                threes.append((-count, move))
        threes.sort()
        return found + [(move, "three") for _, move in threes[:VCT_WIDTH]]

    def try_move(self, move: int, kind: str, depth: int) -> list[int] | None:
        board = self.board
        cells = board.cells
        block = min(wins_after(board, self.attacker, move, set())) if kind == "four" else -1
        cells[move] = self.attacker
        try:
            if kind == "four":
                cells[block] = self.defender
                try:
                    if scan(board, self.defender)[0]:
                        return None
                    rest = self.attack(depth - 1)
                    return None if rest is None else [move, block, *rest]
                finally:
                    cells[block] = EMPTY
            rest = self.defend(depth - 1, move)
            return None if rest is None else [move, *rest]
        finally:
            cells[move] = EMPTY

    def defend(self, depth: int, three: int, counters: int = 0) -> list[int] | None:
        """Defender to move against the open three through `three`.

        Returns the attacker's line if every reply loses, or None. `counters`
        counts the counter-fours already played against this three.
        """
        board = self.board
        cells = board.cells
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise _OutOfTime
        key = (depth, three, counters < VCT_COUNTERS, tuple(cells))
        if key in self.defended:
            return self.defended[key]
        replies = self.replies(three, counters < VCT_COUNTERS)
        if not replies:
            self.defended[key] = []
            return []
        principal = None
        for reply in replies:
            cells[reply] = self.defender
            try:
                their_wins = scan(board, self.defender)[0]
                if their_wins:
                    if len(their_wins) > 1:
                        return None
                    (block,) = their_wins
                    cells[block] = self.attacker
                    try:
                        # A counter-four gave the defender another move. Unless
                        # the three is still open, the chain is broken.
                        if _open_fours_near(board, self.attacker, three):
                            rest = self.defend(depth, three, counters + 1)
                            rest = None if rest is None else [block, *rest]
                        else:
                            rest = None
                    finally:
                        cells[block] = EMPTY
                else:
                    rest = self.attack(depth)
            finally:
                cells[reply] = EMPTY
            if rest is None:
                self.defended[key] = None
                return None
            if principal is None:
                principal = [reply, *rest]
        self.defended[key] = principal
        return principal

    def replies(self, three: int, counters: bool = True) -> list[int]:
        """Blocks that leave the three unable to become an open four, then counter-fours.

        A block that still allows a chain of fours is kept here: the next
        attacking step finds that chain, which is cheaper than checking now.
        """
        board = self.board
        cells = board.cells
        squares: set[int] = set()
        for line in windows(board.size)[1][three]:
            if self.defender not in (cells[i] for i in line):
                squares.update(i for i in line if cells[i] == EMPTY)
        blocks = []
        for square in sorted(squares):
            cells[square] = self.defender
            try:
                if not _open_fours_near(board, self.attacker, three):
                    blocks.append(square)
            finally:
                cells[square] = EMPTY
        if not counters:
            return blocks
        own_wins, own_fours = scan(board, self.defender)
        fours = [
            square
            for square in sorted(own_fours)
            if square not in blocks and wins_after(board, self.defender, square, own_wins)
        ]
        return blocks + fours


def _open_fours_near(board: Board, player: int, move: int) -> set[int]:
    """Open-four moves for `player` in the windows through `move`, a stone already placed."""
    cells = board.cells
    other = opponent(player)
    squares: set[int] = set()
    for line in windows(board.size)[1][move]:
        values = [cells[i] for i in line]
        if other not in values and values.count(player) == 3:
            squares.update(i for i in line if cells[i] == EMPTY)
    return {square for square in squares if len(wins_after(board, player, square, set())) >= 2}


def _open_fours_made(board: Board, player: int, move: int) -> set[int]:
    """Open-four moves `player` would have after playing `move`, found near `move` only."""
    board.cells[move] = player
    try:
        return _open_fours_near(board, player, move)
    finally:
        board.cells[move] = EMPTY


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
