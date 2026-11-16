"""Time-limited alpha-beta search over the pattern evaluator.

Immediate wins and blocks are answered before any search, so a one-move
tactic is never missed because the clock ran out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from engine.board import BLACK, Board, opponent, square_name
from engine.evaluate import WIN_SCORE, evaluate

MAX_DEPTH = 4
DEFAULT_TIME_LIMIT = 1.0
_INF = WIN_SCORE + MAX_DEPTH + 1


@dataclass(frozen=True)
class SearchResult:
    row: int
    col: int
    score: int
    line: list[tuple[int, int]]
    reason: str


class _Timeout(Exception):
    pass


def best_move(
    board: Board,
    side_to_move: int,
    time_limit: float = DEFAULT_TIME_LIMIT,
    max_depth: int = MAX_DEPTH,
) -> SearchResult | None:
    """Return the best move for `side_to_move`, or None when no move exists.

    `score` is from that side's point of view: positive is good for them.
    `line` is the principal variation, starting with the chosen move.
    """
    candidates = board.legal_moves()
    if board.winner or not candidates:
        return None

    own_wins = _winning_squares(board, side_to_move)
    if own_wins:
        row, col = own_wins[0]
        return SearchResult(row, col, WIN_SCORE, [(row, col)], "Takes the win.")

    blocks = _winning_squares(board, opponent(side_to_move))
    if len(blocks) == 1:
        row, col = blocks[0]
        return SearchResult(row, col, 0, [(row, col)], "Blocks the opponent's win.")
    if len(blocks) > 1:
        row, col = blocks[0]
        return SearchResult(
            row,
            col,
            -WIN_SCORE,
            [(row, col)],
            "Opponent has two winning moves. Blocking one of them.",
        )

    if len(candidates) == 1:
        row, col = candidates[0]
        board.place(row, col, side_to_move)
        try:
            score = _perspective(board, side_to_move)
        finally:
            board.undo()
        return SearchResult(row, col, score, [(row, col)], _search_reason(score, [(row, col)]))

    deadline = time.perf_counter() + time_limit
    fallback = _ordered(board, side_to_move)
    best: SearchResult | None = None
    for depth in range(1, max_depth + 1):
        try:
            row, col, score, line = _search_root(board, side_to_move, depth, deadline)
        except _Timeout:
            break
        best = SearchResult(row, col, score, line, _search_reason(score, line))
        if abs(score) >= WIN_SCORE - MAX_DEPTH:
            break
    if best is not None:
        return best
    if not fallback:
        return None
    row, col = fallback[0]
    return SearchResult(row, col, 0, [(row, col)], _search_reason(0, [(row, col)]))


def _winning_squares(board: Board, player: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for row, col in board.legal_moves():
        board.place(row, col, player)
        won = board.winner == player
        board.undo()
        if won:
            found.append((row, col))
    return sorted(found)


def _search_reason(score: int, line: list[tuple[int, int]]) -> str:
    text = ", ".join(square_name(r, c) for r, c in line)
    if score >= WIN_SCORE - MAX_DEPTH:
        return f"Forced win. Line: {text}."
    if score <= -(WIN_SCORE - MAX_DEPTH):
        return f"Forced loss. Line: {text}."
    return f"Score {score:+d}. Line: {text}."


def _search_root(
    board: Board,
    side: int,
    depth: int,
    deadline: float,
) -> tuple[int, int, int, list[tuple[int, int]]]:
    _check(deadline)
    moves = _ordered(board, side)
    best_score = -_INF
    best_row, best_col = moves[0]
    best_line = [moves[0]]
    alpha = -_INF
    for row, col in moves:
        _check(deadline)
        board.place(row, col, side)
        try:
            child: list[tuple[int, int]] = []
            score = -_negamax(
                board,
                depth - 1,
                -_INF,
                -alpha,
                opponent(side),
                child,
                ply=1,
                deadline=deadline,
            )
        finally:
            board.undo()
        if score > best_score:
            best_score = score
            best_row, best_col = row, col
            best_line = [(row, col), *child]
            alpha = score
    return best_row, best_col, best_score, best_line


def _negamax(
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
    side: int,
    pv: list[tuple[int, int]],
    ply: int,
    deadline: float,
) -> int:
    _check(deadline)
    if board.winner:
        return -(WIN_SCORE - ply)
    if depth == 0:
        return _perspective(board, side)
    moves = board.legal_moves()
    if not moves:
        return 0
    moves.sort(key=lambda move: _move_key(board, move))
    best = -_INF
    best_line: list[tuple[int, int]] = []
    for row, col in moves:
        board.place(row, col, side)
        try:
            child: list[tuple[int, int]] = []
            score = -_negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                opponent(side),
                child,
                ply + 1,
                deadline,
            )
        finally:
            board.undo()
        if score > best:
            best = score
            best_line = [(row, col), *child]
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    pv.extend(best_line)
    return best


def _move_key(board: Board, move: tuple[int, int]) -> tuple[int, int, int, int]:
    """Replies next to the last stone first, then moves closer to the center."""
    mid = board.size // 2
    center = abs(move[0] - mid) + abs(move[1] - mid)
    if board.history:
        last_row, last_col, _ = board.history[-1]
        near = max(abs(move[0] - last_row), abs(move[1] - last_col))
    else:
        near = 0
    return (near, center, move[0], move[1])


def _ordered(board: Board, side: int) -> list[tuple[int, int]]:
    ranked: list[tuple[int, int, int, int]] = []
    mid = board.size // 2
    for row, col in board.legal_moves():
        board.place(row, col, side)
        try:
            score = _perspective(board, side)
        finally:
            board.undo()
        ranked.append((score, abs(row - mid) + abs(col - mid), row, col))
    # Higher score first. On a tie, play closer to the center, then by row and column.
    ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [(row, col) for _, _, row, col in ranked]


def _perspective(board: Board, side: int) -> int:
    score = evaluate(board)
    return score if side == BLACK else -score


def _check(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise _Timeout
