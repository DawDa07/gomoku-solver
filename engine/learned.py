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
from engine.evaluate import WIN_SCORE
from engine.net import CHECKPOINT, encode, load_checkpoint
from engine.search import MAX_DEPTH, SearchResult, best_move
from engine.tactics import find_tactic

DEVICE = torch.device("cpu")
BEST = CHECKPOINT.with_name("best.pt")
WHITE_NET = CHECKPOINT.with_name("white.pt")
# Long enough for the search to see a short forced win, short enough for a click.
WHITE_SEARCH_TIME = 0.3
_ATTACK: torch.nn.Module | None = None
_ATTACK_TRIED = False


def _attack_net():
    """Network that plays to win. Falls back to models/best.pt when models/white.pt is absent."""
    global _ATTACK, _ATTACK_TRIED
    if not _ATTACK_TRIED:
        _ATTACK = load_checkpoint(DEVICE, WHITE_NET)
        if _ATTACK is None:
            _ATTACK = load_checkpoint(DEVICE, BEST)
        _ATTACK_TRIED = True
    return _ATTACK


def _from_tactic(tactic) -> SearchResult:
    row, col = tactic.moves[0]
    line = list(tactic.line) or [(row, col)]
    return SearchResult(row, col, tactic.score, line, tactic.reason)


def white_move(board: Board, side: int) -> SearchResult | None:
    """Win if one exists, stall only while the book still proves a loss, otherwise attack.

    A single threat move is played first, including a five, a block, and an open
    four. A short search then takes a win it can see. The book is only the
    fallback when that search does not find a win and the position is still a
    proved loss. Off that line, the attack network plays.
    """
    from engine.book import lookup

    if board.winner or not board.legal_moves():
        return None
    tactic = find_tactic(board, side)
    if tactic is not None and len(tactic.moves) == 1:
        return _from_tactic(tactic)
    searched = best_move(board, side, time_limit=WHITE_SEARCH_TIME)
    if searched is not None and searched.score >= WIN_SCORE - MAX_DEPTH:
        return SearchResult(
            searched.row,
            searched.col,
            searched.score,
            searched.line,
            "Search finds a win. " + searched.reason,
        )
    found = lookup(board, side)
    if found.move is not None:
        return found.move
    return attack_move(board, side)


def attack_move(board: Board, side: int) -> SearchResult | None:
    """Threat checks, then models/white.pt, then models/best.pt."""
    net = _attack_net()
    if net is None:
        result = best_move(board, side)
        if result is None:
            return None
        return SearchResult(
            result.row,
            result.col,
            result.score,
            result.line,
            "No attack checkpoint, so the base bot played. " + result.reason,
        )
    if board.winner or not board.legal_moves():
        return None
    row, col, reason, value = select_action(net, board, side, temperature=0.0)
    if reason.startswith("Learned policy."):
        reason = "Attack policy." + reason[len("Learned policy.") :]
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
