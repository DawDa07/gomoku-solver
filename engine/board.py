"""Freestyle Gomoku board.

The default board is 15×15. A smaller `size` is used while training, because
a full-size game is slow to repeat thousands of times. Five or more stones in
a row wins. Edges do not wrap, and there are no forbidden moves.
"""

from __future__ import annotations

SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

# (row step, column step). The opposite direction is the same line.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def opponent(player: int) -> int:
    if player == BLACK:
        return WHITE
    if player == WHITE:
        return BLACK
    raise ValueError(f"unknown player {player}")


def square_name(row: int, col: int) -> str:
    """Column letter and row number, with row 1 at the top of the board."""
    return f"{chr(ord('A') + col)}{row + 1}"


class Board:
    def __init__(self, cells: list[int] | None = None, size: int = SIZE):
        if size < 5:
            raise ValueError("board must be at least 5")
        self.size = size
        if cells is None:
            self.cells = [EMPTY] * (size * size)
            self.winner = EMPTY
        else:
            if len(cells) != size * size:
                raise ValueError(f"board must have {size * size} cells")
            self.cells = list(cells)
            self.winner = self.find_winner()
        self.history: list[tuple[int, int, int]] = []
        self._winner_before: list[int] = []

    @classmethod
    def from_rows(cls, rows: list[list[int]]) -> Board:
        size = len(rows)
        if size < 5 or any(len(row) != size for row in rows):
            raise ValueError(f"expected a square board of at least 5, got {size}")
        cells: list[int] = []
        for row in rows:
            cells.extend(row)
        return cls(cells, size=size)

    def to_rows(self) -> list[list[int]]:
        return [self.cells[r * self.size : (r + 1) * self.size] for r in range(self.size)]

    def get(self, row: int, col: int) -> int:
        return self.cells[row * self.size + col]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def place(self, row: int, col: int, player: int) -> None:
        if player not in (BLACK, WHITE):
            raise ValueError("player must be black or white")
        if not self.in_bounds(row, col):
            raise ValueError("out of bounds")
        if self.winner:
            raise ValueError("game is over")
        index = row * self.size + col
        if self.cells[index] != EMPTY:
            raise ValueError("cell occupied")
        self._winner_before.append(self.winner)
        self.cells[index] = player
        self.history.append((row, col, player))
        if self._wins_at(row, col, player):
            self.winner = player

    def undo(self) -> tuple[int, int, int] | None:
        if not self.history:
            return None
        row, col, player = self.history.pop()
        self.cells[row * self.size + col] = EMPTY
        self.winner = self._winner_before.pop()
        return row, col, player

    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def legal_moves(self, distance: int = 2) -> list[tuple[int, int]]:
        """Empty cells within `distance` of a stone. The center, if the board is empty.

        A win is always adjacent to an existing stone, so a radius of 2
        still contains every immediately tactical reply.
        """
        if self.winner:
            return []
        stones = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.cells[r * self.size + c] != EMPTY
        ]
        if not stones:
            return [(self.size // 2, self.size // 2)]
        found: set[tuple[int, int]] = set()
        for row, col in stones:
            for dr in range(-distance, distance + 1):
                for dc in range(-distance, distance + 1):
                    rr, cc = row + dr, col + dc
                    if self.in_bounds(rr, cc) and self.cells[rr * self.size + cc] == EMPTY:
                        found.add((rr, cc))
        if not found and not self.is_full():
            return [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if self.cells[r * self.size + c] == EMPTY
            ]
        return sorted(found)

    def _wins_at(self, row: int, col: int, player: int) -> bool:
        for dr, dc in DIRECTIONS:
            count = 1
            r, c = row + dr, col + dc
            while self.in_bounds(r, c) and self.cells[r * self.size + c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while self.in_bounds(r, c) and self.cells[r * self.size + c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False

    def find_winner(self) -> int:
        """Return the player who has five or more in a row, or EMPTY."""
        for row in range(self.size):
            for col in range(self.size):
                player = self.cells[row * self.size + col]
                if player == EMPTY:
                    continue
                for dr, dc in DIRECTIONS:
                    pr, pc = row - dr, col - dc
                    if self.in_bounds(pr, pc) and self.cells[pr * self.size + pc] == player:
                        continue
                    count = 0
                    r, c = row, col
                    while self.in_bounds(r, c) and self.cells[r * self.size + c] == player:
                        count += 1
                        r += dr
                        c += dc
                    if count >= 5:
                        return player
        return EMPTY
