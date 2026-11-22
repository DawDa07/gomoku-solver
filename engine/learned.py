"""Choose a move with the learned network.

The same threat tactics as the base bot run first: win, block a four, make an
open four, play forced fours, block an open three. When a tactic leaves
several safe replies, the network chooses among them. In a quiet position it
chooses among every candidate move.
"""

from __future__ import annotations

import numpy as np
import torch

from engine.board import Board
from engine.net import CHECKPOINT, encode, load_checkpoint
from engine.search import SearchResult, best_move
from engine.tactics import find_tactic

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
    if board.winner or not board.legal_moves():
        return None
    row, col, reason, value = select_action(net, board, side, temperature=0.0)
    return SearchResult(row, col, int(round(value * 100)), [(row, col)], reason)


def select_action(
    net: torch.nn.Module,
    board: Board,
    side: int,
    temperature: float,
) -> tuple[int, int, str, float]:
    """Return row, col, the reason, and the value for the side to move."""
    tactic = find_tactic(board, side)
    if tactic is not None and len(tactic.moves) == 1:
        row, col = tactic.moves[0]
        value = float(np.sign(tactic.score))
        return row, col, tactic.reason, value

    allowed = list(tactic.moves) if tactic is not None else board.legal_moves()
    if not allowed:
        raise ValueError("no moves left")
    logits, value = _infer(net, encode(board, side))
    flat = [row * board.size + col for row, col in allowed]
    chosen = _pick(logits, flat, temperature)
    if tactic is not None:
        reason = (
            f"{tactic.reason} The network chose among {len(allowed)} safe replies. "
            f"Value {value:+.2f}."
        )
    else:
        reason = f"Learned policy. Value {value:+.2f}."
    return chosen // board.size, chosen % board.size, reason, value


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
