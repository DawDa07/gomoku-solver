"""Freestyle Gomoku engine.

`best_move` is the only function the web app needs. A later neural net or
FPGA accelerator can replace the evaluator behind that function.
"""

from engine.board import BLACK, EMPTY, WHITE, Board, opponent, square_name
from engine.search import SearchResult, best_move

__all__ = [
    "BLACK",
    "EMPTY",
    "WHITE",
    "Board",
    "SearchResult",
    "best_move",
    "opponent",
    "square_name",
]
