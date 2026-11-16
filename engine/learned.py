"""Choose a move with the learned network.

A win in one, and a block of a win in one, are still taken from the rules.
The network chooses only when the position is not decided on the next move.
"""

from __future__ import annotations

import numpy as np
import torch

from engine.board import Board, opponent
from engine.net import CHECKPOINT, encode, load_checkpoint
from engine.search import SearchResult, _winning_squares, best_move
from engine.evaluate import WIN_SCORE

DEVICE = torch.device("cpu")
_NET: torch.nn.Module | None = None
_LOADED = False


def _net():
    global _NET, _LOADED
    if not _LOADED:
        _NET = load_checkpoint(DEVICE, CHECKPOINT)
        _LOADED = True
    return _NET


def learned_move(board: Board, side: int) -> SearchResult | None:
    net = _net()
    if net is None:
        result = best_move(board, side)
        if result is None:
            return None
        return SearchResult(
            result.row,
            result.col,
            result.score,
            result.line,
            "No learned checkpoint, so the base bot played. " + result.reason,
        )
    row, col, source, value = select_action(net, board, side, temperature=0.0)
    if source == "win":
        return SearchResult(row, col, WIN_SCORE, [(row, col)], "Takes the win.")
    if source == "block":
        return SearchResult(row, col, 0, [(row, col)], "Blocks the opponent's win.")
    return SearchResult(
        row,
        col,
        int(round(value * 100)),
        [(row, col)],
        f"Learned policy. Value {value:+.2f}.",
    )


def select_action(
    net: torch.nn.Module,
    board: Board,
    side: int,
    temperature: float,
) -> tuple[int, int, str, float]:
    """Return row, col, source (`win`, `block`, or `policy`), and a value."""
    wins = _winning_squares(board, side)
    if wins:
        return wins[0][0], wins[0][1], "win", 1.0
    blocks = _winning_squares(board, opponent(side))
    if blocks:
        return blocks[0][0], blocks[0][1], "block", 0.0

    legal = board.legal_moves()
    if not legal:
        raise ValueError("no moves left")
    logits, value = _infer(net, encode(board, side))
    flat = [row * board.size + col for row, col in legal]
    chosen = _pick(logits, flat, temperature)
    return chosen // board.size, chosen % board.size, "policy", value


def _infer(net: torch.nn.Module, planes: np.ndarray) -> tuple[np.ndarray, float]:
    net.eval()
    with torch.no_grad():
        boards = torch.from_numpy(planes).unsqueeze(0)
        logits, value = net(boards)
    return logits.squeeze(0).numpy(), float(value.item())


def _pick(logits: np.ndarray, legal: list[int], temperature: float) -> int:
    scores = logits[legal]
    if temperature <= 1e-4:
        return legal[int(np.argmax(scores))]
    scaled = scores / temperature
    scaled -= scaled.max()
    weights = np.exp(scaled)
    weights /= weights.sum()
    return int(np.random.choice(legal, p=weights))
