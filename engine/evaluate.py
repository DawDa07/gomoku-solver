"""Pattern score for a freestyle Gomoku position.

Every five-cell window that holds stones of only one color scores for that
color, by how many stones it holds. A solid three and a broken three (XX.X)
both fill windows with three stones, so both count.

`evaluate` reads only the stones on the board. That is the seam a later FPGA
accelerator can sit behind: board in, score out.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from engine.board import BLACK, WHITE, Board
from engine.threats import window_index

# Points for a window holding 0 to 5 stones of one color and none of the other.
WINDOW = np.array([0, 1, 12, 150, 2_500, 0], dtype=np.int64)

WIN_SCORE = 1_000_000


def evaluate(board: Board) -> int:
    """Score from Black's point of view. Positive means Black is ahead."""
    if board.winner == BLACK:
        return WIN_SCORE
    if board.winner == WHITE:
        return -WIN_SCORE
    cells = np.asarray(board.cells, dtype=np.int8)
    values = cells[window_index(board.size)]
    black = (values == BLACK).sum(axis=1)
    white = (values == WHITE).sum(axis=1)
    score = WINDOW[black][white == 0].sum() - WINDOW[white][black == 0].sum()
    stones = (cells == BLACK).astype(np.int64) - (cells == WHITE)
    return int(score + (_center(board.size) * stones).sum())


@lru_cache(maxsize=None)
def _center(size: int) -> np.ndarray:
    """A small nudge toward the middle, so equal shapes prefer the center."""
    mid = size // 2
    return np.array(
        [(size - 1) - (abs(row - mid) + abs(col - mid)) for row in range(size) for col in range(size)],
        dtype=np.int64,
    )
