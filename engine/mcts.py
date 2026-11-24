"""Network-guided tree search, as in AlphaZero.

Each playout walks down the tree, at every step picking the move with the
best mix of average result so far and the network's prior for it (PUCT), then
asks the network about the position it reaches. The network's value is backed
up the path, so moves that keep winning get visited more. The most-visited
move is played, and the visit counts are the training target for self-play.

Several leaves are evaluated in one network call. While a leaf waits, its path
carries a "virtual loss", so the other playouts in the batch look elsewhere.

Cheap threat rules apply inside the tree: a side with a win square takes it,
a side facing a four blocks it, and a side with an open four has won. At the
root, `find_tactic` settles forced positions before any search.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from engine.board import EMPTY, Board, opponent, square_name
from engine.net import encode, load_checkpoint
from engine.search import SearchResult
from engine.tactics import find_tactic
from engine.threats import open_four_moves, scan

C_PUCT = 1.5
BATCH = 8
VIRTUAL_LOSS = 1.0


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Node:
    __slots__ = ("moves", "priors", "visits", "totals", "children", "value")

    def __init__(self) -> None:
        self.moves: np.ndarray | None = None
        self.priors: np.ndarray | None = None
        self.visits: np.ndarray | None = None
        # Sum of results for each move, from the view of the side moving here.
        self.totals: np.ndarray | None = None
        self.children: list[Node | None] = []
        # Known result for the side to move, when threats decide the position.
        self.value: float | None = None

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def expand(self, moves: list[int], priors: np.ndarray) -> None:
        self.moves = np.asarray(moves, dtype=np.int64)
        self.priors = priors.astype(np.float64)
        self.visits = np.zeros(len(moves))
        self.totals = np.zeros(len(moves))
        self.children = [None] * len(moves)

    def select(self) -> int:
        assert self.visits is not None and self.totals is not None and self.priors is not None
        parent = math.sqrt(self.visits.sum() + 1)
        q = np.where(self.visits > 0, self.totals / np.maximum(self.visits, 1), 0.0)
        u = C_PUCT * self.priors * parent / (1 + self.visits)
        return int(np.argmax(q + u))


def _threat_rules(board: Board, side: int) -> tuple[float | None, list[int] | None]:
    """A known result for `side` to move, or the only moves worth searching."""
    wins, fours = scan(board, side)
    if wins:
        return 1.0, None
    their_wins = scan(board, opponent(side))[0]
    if len(their_wins) > 1:
        return -1.0, None
    if their_wins:
        return None, sorted(their_wins)
    if open_four_moves(board, side, wins, fours):
        return 1.0, None
    return None, None


class MCTS:
    def __init__(self, net: torch.nn.Module, device: torch.device, batch: int = BATCH):
        self.net = net
        self.device = device
        self.batch = batch

    def _evaluate(self, boards: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            tensor = torch.from_numpy(np.stack(boards)).to(self.device)
            logits, values = self.net(tensor)
        return logits.float().cpu().numpy(), values.float().cpu().numpy()

    def _priors(self, logits: np.ndarray, moves: list[int]) -> np.ndarray:
        scores = logits[moves].astype(np.float64)
        scores -= scores.max()
        weights = np.exp(scores)
        return weights / weights.sum()

    def search(
        self,
        board: Board,
        side: int,
        simulations: int,
        root_moves: list[int] | None = None,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run playouts from `board`. Returns the root moves, their visit counts, and the root value.

        `noise` mixes that share of Dirichlet noise into the root priors, so
        self-play tries moves the network does not yet favor.
        """
        root = Node()
        work = Board(list(board.cells), size=board.size)
        moves = root_moves or [row * board.size + col for row, col in board.legal_moves()]
        logits, values = self._evaluate([encode(work, side)])
        priors = self._priors(logits[0], moves)
        if noise > 0:
            rng = rng or np.random.default_rng()
            priors = (1 - noise) * priors + noise * rng.dirichlet([0.3] * len(moves))
        root.expand(moves, priors)
        root_value = float(values[0])

        done = 0
        while done < simulations:
            pending: list[tuple[list[tuple[Node, int]], Node, int]] = []
            planes: list[np.ndarray] = []
            for _ in range(min(self.batch, simulations - done)):
                path, leaf, leaf_side = self._descend(root, work, side)
                if leaf.value is not None:
                    self._backup(path, leaf.value)
                elif work.winner:
                    leaf.value = -1.0
                    self._backup(path, -1.0)
                elif EMPTY not in work.cells:
                    leaf.value = 0.0
                    self._backup(path, 0.0)
                else:
                    known, only = _threat_rules(work, leaf_side)
                    if known is not None:
                        leaf.value = known
                        self._backup(path, known)
                    else:
                        legal = only or [r * work.size + c for r, c in work.legal_moves()]
                        leaf.expand(legal, np.full(len(legal), 1.0 / len(legal)))
                        pending.append((path, leaf, leaf_side))
                        planes.append(encode(work, leaf_side))
                        self._add_virtual_loss(path)
                for _ in path:
                    work.undo()
                done += 1
            if planes:
                logits, values = self._evaluate(planes)
                for (path, leaf, _), logit, value in zip(pending, logits, values):
                    assert leaf.moves is not None
                    leaf.priors = self._priors(logit, leaf.moves.tolist())
                    self._remove_virtual_loss(path)
                    self._backup(path, float(value))
        assert root.visits is not None
        return root.moves, root.visits.copy(), root_value

    def _descend(self, root: Node, work: Board, side: int) -> tuple[list[tuple[Node, int]], Node, int]:
        """Walk to a leaf, placing stones on `work`. Returns the path, the leaf, and the side to move there."""
        node = root
        path: list[tuple[Node, int]] = []
        while node.expanded and node.value is None and node.moves is not None and len(node.moves):
            index = node.select()
            move = int(node.moves[index])
            work.place(move // work.size, move % work.size, side)
            path.append((node, index))
            child = node.children[index]
            if child is None:
                child = Node()
                node.children[index] = child
            node = child
            side = opponent(side)
            if work.winner:
                break
        return path, node, side

    def _backup(self, path: list[tuple[Node, int]], value: float) -> None:
        """`value` is for the side to move at the leaf; each step up flips it."""
        for node, index in reversed(path):
            value = -value
            assert node.visits is not None and node.totals is not None
            node.visits[index] += 1
            node.totals[index] += value

    def _add_virtual_loss(self, path: list[tuple[Node, int]]) -> None:
        for node, index in path:
            assert node.visits is not None and node.totals is not None
            node.visits[index] += VIRTUAL_LOSS
            node.totals[index] -= VIRTUAL_LOSS

    def _remove_virtual_loss(self, path: list[tuple[Node, int]]) -> None:
        for node, index in path:
            assert node.visits is not None and node.totals is not None
            node.visits[index] -= VIRTUAL_LOSS
            node.totals[index] += VIRTUAL_LOSS


class MCTSPlayer:
    """Tree search with a trained network, `find_tactic` first."""

    def __init__(
        self,
        path: Path,
        simulations: int = 200,
        name: str = "mcts",
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        net = load_checkpoint(self.device, path)
        if net is None:
            raise FileNotFoundError(f"no checkpoint at {path}")
        self.search = MCTS(net, self.device)
        self.simulations = simulations
        self.name = name

    def move(self, board: Board, side: int) -> SearchResult | None:
        if board.winner or not board.legal_moves():
            return None
        return mcts_move(self.search, board, side, self.simulations)

    def close(self) -> None:
        pass


def mcts_move(search: MCTS, board: Board, side: int, simulations: int) -> SearchResult:
    tactic = find_tactic(board, side)
    if tactic is not None and len(tactic.moves) == 1:
        row, col = tactic.moves[0]
        return SearchResult(row, col, tactic.score, list(tactic.line) or [(row, col)], tactic.reason)
    roots = [row * board.size + col for row, col in tactic.moves] if tactic is not None else None
    moves, visits, value = search.search(board, side, simulations, roots)
    best = int(moves[int(np.argmax(visits))])
    row, col = divmod(best, board.size)
    share = visits.max() / max(visits.sum(), 1)
    text = f"Tree search: {int(visits.sum())} playouts, {share:.0%} on {square_name(row, col)}, value {value:+.2f}."
    reason = f"{tactic.reason} {text}" if tactic is not None else text
    return SearchResult(row, col, int(round(value * 100)), [(row, col)], reason)
