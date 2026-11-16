"""Pattern scores for a freestyle Gomoku position.

`evaluate` reads only the stones on the board (and the win that follows from
them). It does not depend on move history, which is the seam a later FPGA
accelerator can sit behind: board in, score out.
"""

from __future__ import annotations

from engine.board import BLACK, DIRECTIONS, EMPTY, WHITE, Board

# Tuned so one forcing pattern outranks a pile of quiet shapes, and a
# finished five outranks every threat. Two open fours add up, so a double
# threat beats a single one.
FIVE = 100_000
OPEN_FOUR = 50_000
CLOSED_FOUR = 8_000
OPEN_THREE = 3_000
CLOSED_THREE = 400
OPEN_TWO = 80
CLOSED_TWO = 15

WIN_SCORE = 1_000_000


def evaluate(board: Board) -> int:
    """Score from Black's point of view. Positive means Black is ahead."""
    if board.winner == BLACK:
        return WIN_SCORE
    if board.winner == WHITE:
        return -WIN_SCORE
    return _pattern_score(board, BLACK) - _pattern_score(board, WHITE) + _center(board)


def _pattern_score(board: Board, player: int) -> int:
    total = 0
    cells = board.cells
    size = board.size
    for row in range(size):
        for col in range(size):
            if cells[row * size + col] != player:
                continue
            for dr, dc in DIRECTIONS:
                pr, pc = row - dr, col - dc
                if board.in_bounds(pr, pc) and cells[pr * size + pc] == player:
                    continue
                length = 0
                r, c = row, col
                while board.in_bounds(r, c) and cells[r * size + c] == player:
                    length += 1
                    r += dr
                    c += dc
                open_ends = 0
                if board.in_bounds(pr, pc) and cells[pr * size + pc] == EMPTY:
                    open_ends += 1
                if board.in_bounds(r, c) and cells[r * size + c] == EMPTY:
                    open_ends += 1
                total += _run_score(length, open_ends)
    return total


def _run_score(length: int, open_ends: int) -> int:
    if length >= 5:
        return FIVE
    if open_ends == 0:
        return 0
    if length == 4:
        return OPEN_FOUR if open_ends == 2 else CLOSED_FOUR
    if length == 3:
        return OPEN_THREE if open_ends == 2 else CLOSED_THREE
    if length == 2:
        return OPEN_TWO if open_ends == 2 else CLOSED_TWO
    return 1 if open_ends == 2 else 0


def _center(board: Board) -> int:
    """A small nudge toward the middle, so equal shapes prefer the center."""
    bonus = 0
    size = board.size
    mid = size // 2
    for row in range(size):
        for col in range(size):
            player = board.cells[row * size + col]
            if player == EMPTY:
                continue
            weight = (size - 1) - (abs(row - mid) + abs(col - mid))
            bonus += weight if player == BLACK else -weight
    return bonus
