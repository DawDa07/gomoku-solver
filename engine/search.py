"""Threat tactics first, then a time-limited alpha-beta search.

`find_tactic` settles positions decided by threats: wins, blocks, open fours,
forced fours, chains of threes, double threats, and open threes that must be
stopped. When it allows several safe replies, alpha-beta chooses among only
those.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from engine.board import BLACK, Board, opponent, square_name
from engine.evaluate import WIN_SCORE, evaluate
from engine.tactics import find_tactic

MAX_DEPTH = 4
DEFAULT_TIME_LIMIT = 1.0
# Share of the move time the threat checks may use before alpha-beta.
TACTIC_SHARE = 0.4
# Stopping the opponent's chain of threes may use more, since missing it loses.
DEFENSE_SHARE = 0.85
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
    time_limit: float | None = DEFAULT_TIME_LIMIT,
    max_depth: int = MAX_DEPTH,
) -> SearchResult | None:
    """Return the best move for `side_to_move`, or None when no move exists.

    With `time_limit=None` the search runs every depth up to `max_depth`, so
    the same position always gives the same move.

    `score` is from that side's point of view: positive is good for them.
    `line` is the principal variation, starting with the chosen move.
    """
    candidates = board.legal_moves()
    if board.winner or not candidates:
        return None

    started = time.perf_counter()
    if time_limit is None:
        deadline = math.inf
        tactic_deadline = defense_deadline = None
    else:
        deadline = started + time_limit
        tactic_deadline = started + TACTIC_SHARE * time_limit
        defense_deadline = started + DEFENSE_SHARE * time_limit
    tactic = find_tactic(board, side_to_move, tactic_deadline, defense_deadline)
    if tactic is not None and len(tactic.moves) == 1:
        row, col = tactic.moves[0]
        line = list(tactic.line) or [(row, col)]
        return SearchResult(row, col, tactic.score, line, tactic.reason)

    roots = list(tactic.moves) if tactic is not None else candidates
    prefix = f"{tactic.reason} " if tactic is not None else ""
    best: SearchResult | None = None
    for depth in range(1, max_depth + 1):
        try:
            row, col, score, line = _search_root(board, side_to_move, depth, deadline, roots)
        except _Timeout:
            break
        best = SearchResult(row, col, score, line, prefix + _search_reason(score, line))
        if abs(score) >= WIN_SCORE - MAX_DEPTH:
            break
    if best is not None:
        return best
    row, col = _ordered(board, side_to_move, roots)[0]
    return SearchResult(row, col, 0, [(row, col)], prefix + _search_reason(0, [(row, col)]))


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
    roots: list[tuple[int, int]],
) -> tuple[int, int, int, list[tuple[int, int]]]:
    _check(deadline)
    moves = _ordered(board, side, roots)
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


def _ordered(board: Board, side: int, moves: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ranked: list[tuple[int, int, int, int]] = []
    mid = board.size // 2
    for row, col in moves:
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
